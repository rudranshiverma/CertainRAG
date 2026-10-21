import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from certainrag.signals import RetrievalSignal
from evaluation import data

RESULTS_DIR = Path("evaluation/results/retrieval")
EMBEDDERS = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small-v2": ("intfloat/e5-small-v2", "query: ", "passage: "),
}
LIMITS = {"train": 300, "dev": 150, "test": 400}


def with_cross_domain_negatives(queries_by_subset):
    # pair each question with documents from a query in a distant subset
    subsets = list(queries_by_subset)
    if len(subsets) < 2:
        raise ValueError("cross-domain negatives need at least two subsets")
    shift = len(subsets) // 2
    out = []
    for i, subset in enumerate(subsets):
        others = queries_by_subset[subsets[(i + shift) % len(subsets)]]
        for j, q in enumerate(queries_by_subset[subset]):
            out.append({**q, "kind": "original"})
            donor = others[j % len(others)]
            out.append(
                {
                    **q,
                    "id": f"{q['id']}-x-{donor['id']}",
                    "kind": "cross_domain",
                    "documents": donor["documents"],
                    "relevant": [0] * len(donor["documents"]),
                }
            )
    return out


class Scorers:
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer

        self.reranker = RetrievalSignal(device=device)
        self.embedders = {
            name: (SentenceTransformer(checkpoint, device=device), q_prefix, d_prefix)
            for name, (checkpoint, q_prefix, d_prefix) in EMBEDDERS.items()
        }

    def __call__(self, query):
        start = time.perf_counter()
        logits = self.reranker.logits(query["question"], query["documents"])
        latency = time.perf_counter() - start
        cosine = {}
        for name, (model, q_prefix, d_prefix) in self.embedders.items():
            q = model.encode([q_prefix + query["question"]], normalize_embeddings=True)
            d = model.encode([d_prefix + doc for doc in query["documents"]], normalize_embeddings=True)
            cosine[name] = np.round(d @ q[0], 6).tolist()
        return {"reranker": np.round(logits, 6).tolist(), "cosine": cosine, "latency": latency}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=list(LIMITS))
    parser.add_argument("--subsets", nargs="+", default=data.RAGBENCH_SUBSETS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scorers = Scorers(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        queries = {s: data.ragbench_retrieval(s, split, limit=LIMITS[split]) for s in args.subsets}
        path = RESULTS_DIR / f"{split}.jsonl"
        done = set()
        if path.exists():
            with open(path) as f:
                done = {json.loads(line)["id"] for line in f}
        todo = [q for q in with_cross_domain_negatives(queries) if q["id"] not in done]
        print(f"{path}: {len(done)} cached, {len(todo)} to run")
        with open(path, "a") as f:
            for q in todo:
                row = {k: q[k] for k in ("id", "group", "kind", "relevant")}
                f.write(json.dumps({**row, **scorers(q)}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
