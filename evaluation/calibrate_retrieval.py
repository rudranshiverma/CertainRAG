import argparse
import json
from pathlib import Path

from certainrag.calibration import fit_platt
from evaluation.run_retrieval import RESULTS_DIR


def chunk_pairs(path):
    logits, labels = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "original":
                logits.extend(row["reranker"])
                labels.extend(row["relevant"])
    return logits, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logits, labels = chunk_pairs(RESULTS_DIR / f"{args.split}.jsonl")
    a, b = fit_platt(logits, labels)
    if a <= 0:
        raise ValueError(f"fitted slope {a:.4f} is not positive; reranker logits do not rank relevance")
    out = Path(RESULTS_DIR / "platt.json")
    out.write_text(json.dumps({"a": a, "b": b, "pairs": len(labels), "positive_rate": sum(labels) / len(labels)}))
    print(f"fitted on {len(labels)} (question, chunk) pairs, {sum(labels) / len(labels):.1%} relevant")
    print(f"DEFAULT_PLATT = ({a:.4f}, {b:.4f})")


if __name__ == "__main__":
    main()
