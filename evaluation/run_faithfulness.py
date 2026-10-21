import argparse
import json
import time
from pathlib import Path

import torch

from certainrag.exceptions import BackendError
from certainrag.signals import VERIFIERS, FaithfulnessSignal, LLMJudge, Verifier, build_claims
from evaluation import data
from evaluation.baselines import HHEM, LETTUCEDETECT_CHECKPOINTS, LettuceDetect

RESULTS_DIR = Path("evaluation/results/faithfulness")


class VerifierMethod:
    def __init__(self, name, device):
        self.signal = FaithfulnessSignal(verifier=Verifier(name, device=device))

    def __call__(self, record):
        if record.get("claim_level"):
            claims = [record["answer"]]
        else:
            claims = build_claims(record["question"], record["answer"])
        if not claims:
            return {"score": 0.0, "claims": [], "matrix": []}
        claims, matrix = self.signal.support_matrix(claims, record["chunks"])
        score = float(matrix.max(axis=1).min()) if matrix.size else 0.0
        return {"score": score, "claims": claims, "matrix": matrix.round(5).tolist()}


class JudgeMethod:
    def __init__(self, model, host, num_ctx):
        import requests

        self.session = requests.Session()
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx
        self.truncated = False
        self.judge = LLMJudge(self._generate)

    def _generate(self, prompt):
        response = self.session.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx},
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        # ollama silently drops the start of prompts longer than num_ctx
        self.truncated = body.get("prompt_eval_count", 0) >= self.num_ctx
        return body["response"]

    def __call__(self, record):
        try:
            score, reasoning = self.judge.score(record["question"], record["answer"], record["chunks"])
        except BackendError as e:
            return {"score": None, "error": str(e)}
        out = {"score": score, "reasoning": reasoning, "truncated": self.truncated}
        if score is None:
            out["error"] = "unparseable"
        return out


def build_method(args):
    if args.method in VERIFIERS:
        return VerifierMethod(args.method, args.device)
    if args.method == "hhem":
        return HHEM(args.device)
    if args.method in LETTUCEDETECT_CHECKPOINTS:
        return LettuceDetect(args.method, args.device)
    if args.method == "judge":
        return JudgeMethod(args.judge_model, args.ollama_host, args.num_ctx)
    raise ValueError(f"Unknown method '{args.method}'")


def method_name(args):
    return f"judge-{args.judge_model.replace(':', '-')}" if args.method == "judge" else args.method


def run(method, records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [r for r in records if r["id"] not in done]
    print(f"{path}: {len(done)} cached, {len(todo)} to run")
    with open(path, "a") as f:
        for i, record in enumerate(todo, 1):
            start = time.perf_counter()
            try:
                out = method(record)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out = {"score": None, "error": "oom"}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row = {"id": record["id"], "group": record["group"], "label": record["label"], **out}
            row["latency"] = time.perf_counter() - start
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--datasets", nargs="+", default=data.FAITHFULNESS_DATASETS)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--limit", type=int, default=400, help="test examples per dataset, 0 for all")
    parser.add_argument("--dev-limit", type=int, default=150, help="dev examples per dataset, 0 for all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--judge-model", default="phi4-mini")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    args = parser.parse_args()

    method = build_method(args)
    for dataset in args.datasets:
        for split in args.splits:
            limit = args.dev_limit if split == "dev" else args.limit
            records = data.load(dataset, split, limit=limit or None)
            run(method, records, RESULTS_DIR / method_name(args) / f"{dataset}_{split}.jsonl")


if __name__ == "__main__":
    main()
