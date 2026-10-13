
import argparse
import json
import re
import urllib.request
from pathlib import Path

import nltk
import torch
import transformers

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
PER_GROUP = 5
MINICHECK_CHUNK_SIZE = 400


def load_jsonl(name, cache_dir):
    path = Path(cache_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAGTRUTH_URL.format(name), path)
    with open(path) as f:
        return [json.loads(line) for line in f]


def first_sentence(text):
    line = text.strip().split("\n")[0]
    match = re.match(r"(.+?[.!?])(\s|$)", line)
    return match.group(1) if match else line


def select_pairs(cache_dir):
    responses = load_jsonl("response.jsonl", cache_dir)
    sources = {s["source_id"]: s for s in load_jsonl("source_info.jsonl", cache_dir)}
    groups = {(t, h): [] for t in ("QA", "Summary") for h in (True, False)}
    for r in sorted(responses, key=lambda r: int(r["id"])):
        src = sources[r["source_id"]]
        key = (src["task_type"], bool(r["labels"]))
        if r["split"] != "test" or r["quality"] != "good" or key not in groups:
            continue
        if len(groups[key]) == PER_GROUP:
            continue
        info = src["source_info"]
        doc = info["passages"] if src["task_type"] == "QA" else info
        claim = r["labels"][0]["text"].strip() if r["labels"] else first_sentence(r["response"])
        if claim:
            groups[key].append({"id": r["id"], "doc": doc, "claim": claim})
    return [p for group in groups.values() for p in group]


def minicheck_rows(model_name, pairs):
    from minicheck.inference import Inferencer

    inf = Inferencer(model_name=model_name, max_model_len=None, batch_size=16, cache_dir=None)
    inf.chunk_size = MINICHECK_CHUNK_SIZE
    eos = inf.tokenizer.eos_token
    captured = {}
    original = inf.inference

    def spy(doc, claim):
        captured["chunks"] = list(doc)
        return original(doc, claim)

    inf.inference = spy
    rows = []
    for p in pairs:
        out = inf.inference_per_example(p["doc"], p["claim"])
        chunks = captured["chunks"]
        if any(len(inf.tokenizer(c + eos + p["claim"])["input_ids"]) > inf.max_model_len for c in chunks):
            continue
        rows.append({"id": p["id"], "chunks": chunks, "probs": out["support_prob_per_chunk"].tolist()})
    return rows


def factcg_rows(pairs):
    from factcg.inference import Inferencer
    from factcg.utils import INSTRUCTION_TEMPLATE

    inf = Inferencer(use_hf_ckpt=True, verbose=False)
    rows = []
    for p in pairs:
        chunks = inf.chunking_src(p["doc"])
        texts = [INSTRUCTION_TEMPLATE.format(text_a=c, text_b=p["claim"]) for c in chunks]
        if any(len(inf.tokenizer(t)["input_ids"]) > 2048 for t in texts):
            continue
        probs = inf.inference(chunks, [p["claim"]] * len(chunks)).tolist()
        rows.append({"id": p["id"], "chunks": chunks, "probs": probs})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/fixtures/parity.json")
    parser.add_argument("--cache-dir", default="evaluation/results/ragtruth")
    args = parser.parse_args()
    # minicheck only fetches "punkt", but recent nltk sent_tokenize needs "punkt_tab"
    nltk.download("punkt_tab", quiet=True)

    pairs = select_pairs(args.cache_dir)
    models = {
        "minicheck-deberta": minicheck_rows("deberta-v3-large", pairs),
        "minicheck-roberta": minicheck_rows("roberta-large", pairs),
        "factcg": factcg_rows(pairs),
    }
    fixture = {
        "meta": {"torch": torch.__version__, "transformers": transformers.__version__},
        "pairs": pairs,
        "models": models,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1))
    for name, rows in models.items():
        n_chunks = sum(len(r["chunks"]) for r in rows)
        multi = sum(len(r["chunks"]) > 1 for r in rows)
        print(f"{name}: {len(rows)}/{len(pairs)} pairs, {n_chunks} windows, {multi} multi-window")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
