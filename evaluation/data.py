import json
import random
import re
import urllib.request
from pathlib import Path

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
RAGTRUTH_TASKS = {"qa": "QA", "summary": "Summary"}
RAGBENCH_SUBSETS = ["pubmedqa", "covidqa", "techqa", "emanual", "cuad", "delucionqa"]
RAGBENCH_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
AGGREFACT_SPLITS = {"dev": "dev", "test": "test"}
CACHE_DIR = Path("evaluation/results/data")


def _sample(items, limit, seed):
    # a seeded shuffle keeps smaller samples nested inside larger ones
    if limit is None or len(items) <= limit:
        return items
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def _ragtruth_files():
    files = {}
    for name in ("response.jsonl", "source_info.jsonl"):
        path = CACHE_DIR / "ragtruth" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RAGTRUTH_URL.format(name), path)
        with open(path) as f:
            files[name] = [json.loads(line) for line in f]
    return files["response.jsonl"], {s["source_id"]: s for s in files["source_info.jsonl"]}


def ragtruth(task, split, dev_sources=100, seed=0):
    responses, sources = _ragtruth_files()
    task_type = RAGTRUTH_TASKS[task]
    rows = [r for r in responses if sources[r["source_id"]]["task_type"] == task_type]
    if split == "test":
        rows = [r for r in rows if r["split"] == "test"]
    elif split == "dev":
        # dev is carved from train by source so no source appears in both
        train_ids = sorted({r["source_id"] for r in rows if r["split"] == "train"}, key=int)
        chosen = set(_sample(train_ids, dev_sources, seed))
        rows = [r for r in rows if r["split"] == "train" and r["source_id"] in chosen]
    else:
        raise ValueError(f"Unknown RAGTruth split '{split}'")
    records = []
    for r in sorted(rows, key=lambda r: int(r["id"])):
        src = sources[r["source_id"]]
        if task_type == "QA":
            question = src["source_info"]["question"]
            chunks = [p.strip() for p in re.split(r"passage \d+:", src["source_info"]["passages"]) if p.strip()]
        else:
            question = src["prompt"].split("\n")[0]
            chunks = [src["source_info"]]
        records.append(
            {
                "id": r["id"],
                "group": f"ragtruth-{task}",
                "question": question,
                "answer": r["response"],
                "chunks": chunks,
                "prompt": src["prompt"],
                "label": int(not r["labels"]),
            }
        )
    return records


def _ragbench(subset, split):
    from datasets import load_dataset

    return load_dataset("rungalileo/ragbench", subset, split=RAGBENCH_SPLITS[split])


def ragbench(subset, split, limit=None, seed=0):
    rows = [r for r in _ragbench(subset, split) if r["adherence_score"] is not None]
    records = [
        {
            "id": r["id"],
            "group": f"ragbench-{subset}",
            "question": r["question"],
            "answer": r["response"],
            "chunks": list(r["documents"]),
            "label": int(r["adherence_score"]),
        }
        for r in rows
    ]
    return _sample(records, limit, seed)


def ragbench_retrieval(subset, split, limit=None, seed=0):
    queries, skipped = [], 0
    for r in _ragbench(subset, split):
        documents = list(r["documents"])
        relevant = {int(re.match(r"\d+", key).group()) for key in r["all_relevant_sentence_keys"]}
        if not relevant <= set(range(len(documents))):
            skipped += 1
            continue
        queries.append(
            {
                "id": r["id"],
                "group": subset,
                "question": r["question"],
                "documents": documents,
                "relevant": [int(i in relevant) for i in range(len(documents))],
            }
        )
    if skipped:
        print(f"ragbench-{subset} {split}: skipped {skipped} rows whose relevance keys point outside their documents")
    return _sample(queries, limit, seed)


def aggrefact(split, limit_per_dataset=None, seed=0):
    from datasets import load_dataset

    rows = load_dataset("lytang/LLM-AggreFact", split=AGGREFACT_SPLITS[split])
    by_dataset = {}
    for i, r in enumerate(rows):
        by_dataset.setdefault(r["dataset"], []).append(
            {
                "id": f"{r['dataset']}-{i}",
                "group": r["dataset"],
                "question": "",
                "answer": r["claim"],
                "chunks": [r["doc"]],
                "label": int(r["label"]),
                "claim_level": True,
            }
        )
    return [rec for recs in by_dataset.values() for rec in _sample(recs, limit_per_dataset, seed)]


def load(name, split, limit=None, seed=0):
    if name.startswith("ragtruth-"):
        return ragtruth(name.removeprefix("ragtruth-"), split, seed=seed)
    if name.startswith("ragbench-"):
        return ragbench(name.removeprefix("ragbench-"), split, limit, seed)
    if name == "aggrefact":
        return aggrefact(split, limit, seed)
    raise ValueError(f"Unknown dataset '{name}'")


FAITHFULNESS_DATASETS = [f"ragtruth-{t}" for t in RAGTRUTH_TASKS] + [f"ragbench-{s}" for s in RAGBENCH_SUBSETS]
