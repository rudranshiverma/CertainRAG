import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

import torch

from certainrag.signals import VERIFIERS, FaithfulnessSignal, RetrievalSignal

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
WARMUP = 3


def load_jsonl(name, cache_dir):
    path = Path(cache_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAGTRUTH_URL.format(name), path)
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_queries(n, cache_dir):
    sources = {s["source_id"]: s for s in load_jsonl("source_info.jsonl", cache_dir)}
    queries = []
    for r in sorted(load_jsonl("response.jsonl", cache_dir), key=lambda r: int(r["id"])):
        src = sources[r["source_id"]]
        if r["split"] != "test" or src["task_type"] != "QA" or r["quality"] != "good":
            continue
        passages = [p.strip() for p in re.split(r"passage \d+:", src["source_info"]["passages"]) if p.strip()]
        queries.append((src["source_info"]["question"], r["response"], passages))
        if len(queries) == n + WARMUP:
            break
    return queries


def percentiles(values):
    values = sorted(values)
    return {
        "p50": statistics.median(values),
        "p95": values[min(len(values) - 1, round(0.95 * (len(values) - 1)))],
        "mean": statistics.fmean(values),
    }


def measure(fn, queries):
    times = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        fn(*query)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.perf_counter() - start)
    return percentiles(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--verifiers", nargs="+", default=list(VERIFIERS))
    parser.add_argument("--cache-dir", default="evaluation/results/ragtruth")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    queries = load_queries(args.n, args.cache_dir)

    retrieval = RetrievalSignal(device=device)
    report = {
        "device": torch.cuda.get_device_name() if device == "cuda" else f"cpu ({torch.get_num_threads()} threads)",
        "queries": len(queries) - WARMUP,
        "retrieval": measure(lambda q, a, c: retrieval.score(q, c), queries),
    }
    for name in args.verifiers:
        signal = FaithfulnessSignal(verifier=name, device=device)
        report[name] = measure(signal.score, queries)
        del signal
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"{report['device']}, {report['queries']} RAGTruth QA queries, seconds per query")
    print(f"{'stage':<20}{'p50':>8}{'p95':>8}{'mean':>8}")
    for stage, stats in report.items():
        if isinstance(stats, dict):
            print(f"{stage:<20}{stats['p50']:>8.3f}{stats['p95']:>8.3f}{stats['mean']:>8.3f}")
    out = Path(args.out or f"evaluation/results/latency_{device}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
