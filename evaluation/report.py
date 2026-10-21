import json
from pathlib import Path

import numpy as np

from certainrag.calibration import apply_platt
from certainrag.signals import VERIFIERS
from evaluation import data, metrics

RESULTS = Path("evaluation/results")
OUT = Path("evaluation/outputs")
BANDS = [(0.35, 0.65), (0.25, 0.75), (0.15, 0.85)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def arrays(rows):
    # failed or unparseable scores count as uninformative rather than being dropped
    labels = np.array([r["label"] for r in rows])
    scores = np.array([0.5 if r["score"] is None else r["score"] for r in rows])
    return labels, scores, sum(r["score"] is None for r in rows)


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def methods():
    root = RESULTS / "faithfulness"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def evaluate(test, dev):
    labels, scores, errors = arrays(test)
    if labels.min() == labels.max():
        return None
    out = {"n": len(test), "errors": errors, "auroc": metrics.auroc(labels, scores)}
    out["ci"] = metrics.bootstrap_ci(metrics.auroc, labels, scores)
    out["p50"], out["p95"] = metrics.percentiles([r["latency"] for r in test])
    out["bacc"] = out["f1"] = None
    if dev:
        dev_labels, dev_scores, _ = arrays(dev)
        if dev_labels.min() != dev_labels.max():
            t = metrics.tuned_threshold(dev_labels, dev_scores)
            out["bacc"] = metrics.balanced_accuracy(labels, scores, t)
            out["f1"] = metrics.unsupported_prf(labels, scores, t)[2]
    return out


def faithfulness_section():
    results = {}
    for method in methods():
        if method.startswith("judge-"):
            continue
        for dataset in data.FAITHFULNESS_DATASETS:
            base = RESULTS / "faithfulness" / method
            test = read(base / f"{dataset}_test.jsonl")
            if test:
                results[method, dataset] = evaluate(test, read(base / f"{dataset}_dev.jsonl"))
    rows, summary = [], {}
    for method in sorted({m for m, _ in results}):
        cells = [results.get((method, d)) for d in data.FAITHFULNESS_DATASETS]
        done = [c for c in cells if c]
        if not done:
            continue
        macro = float(np.mean([c["auroc"] for c in done]))
        p95 = float(np.mean([c["p95"] for c in done]))
        summary[method] = {"auroc": macro, "p95": p95, "complete": len(done) == len(cells)}
        rows.append([method] + [fmt(c["auroc"]) if c else "—" for c in cells] + [fmt(macro), fmt(p95, 2)])
    header = ["method"] + data.FAITHFULNESS_DATASETS + ["macro AUROC", "mean p95 s"]
    detail = []
    for (method, dataset), c in sorted(results.items()):
        if c:
            detail.append(
                [
                    method,
                    dataset,
                    c["n"],
                    f"{c['auroc']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]",
                    fmt(c["bacc"]),
                    fmt(c["f1"]),
                    f"{c['p50']:.2f} / {c['p95']:.2f}",
                    c["errors"],
                ]
            )
    detail_header = ["method", "dataset", "n", "AUROC [95% CI]", "BAcc", "F1 (unsupported)", "p50 / p95 s", "errors"]
    text = "## Faithfulness: AUROC on test\n\n" + table(header, rows)
    text += "\n\n### Per dataset\n\n" + table(detail_header, detail)
    return text, summary


def aggregation_section():
    rows = []
    for method in methods():
        if method.startswith("judge-"):
            continue
        for dataset in data.FAITHFULNESS_DATASETS:
            test = read(RESULTS / "faithfulness" / method / f"{dataset}_test.jsonl")
            if not test or "matrix" not in test[0]:
                continue
            labels = np.array([r["label"] for r in test])
            claim_scores = [np.max(r["matrix"], axis=1) if r["matrix"] and r["matrix"][0] else None for r in test]
            minimum = [s.min() if s is not None else 0.0 for s in claim_scores]
            mean = [s.mean() if s is not None else 0.0 for s in claim_scores]
            if labels.min() != labels.max():
                rows.append([method, dataset, fmt(metrics.auroc(labels, minimum)), fmt(metrics.auroc(labels, mean))])
    return "## Claim aggregation: AUROC with min vs mean over sentences\n\n" + table(
        ["method", "dataset", "min", "mean"], rows
    )


def aggrefact_section():
    rows = []
    for method in methods():
        base = RESULTS / "faithfulness" / method
        test, dev = read(base / "aggrefact_test.jsonl"), read(base / "aggrefact_dev.jsonl")
        if not test:
            continue
        groups = sorted({r["group"] for r in test})
        fixed, tuned = [], []
        for g in groups:
            labels, scores, _ = arrays([r for r in test if r["group"] == g])
            fixed.append(metrics.balanced_accuracy(labels, scores, 0.5, strict=True))
            g_dev = [r for r in dev if r["group"] == g]
            if g_dev:
                dev_labels, dev_scores, _ = arrays(g_dev)
                t = metrics.tuned_threshold(dev_labels, dev_scores)
                tuned.append(metrics.balanced_accuracy(labels, scores, t))
        avg_tuned = fmt(100 * np.mean(tuned), 1) if len(tuned) == len(groups) else "—"
        rows.append([method, len(test), fmt(100 * np.mean(fixed), 1), avg_tuned])
        rows += [[f"  {g}", "", fmt(100 * f, 1), ""] for g, f in zip(groups, fixed, strict=True)]
    header = ["method", "n", "BAcc @0.5", "BAcc @dev threshold"]
    return "## LLM-AggreFact: balanced accuracy (%)\n\n" + table(header, rows)


def cascade_section():
    rows = []
    judges = [m for m in methods() if m.startswith("judge-")]
    verifiers = [m for m in methods() if m in VERIFIERS]
    for judge in judges:
        for verifier in verifiers:
            for family, prefix in [("RAGTruth-QA", "ragtruth-qa"), ("RAGBench", "ragbench-")]:
                j_rows, v_rows = {}, {}
                for dataset in data.FAITHFULNESS_DATASETS:
                    if dataset.startswith(prefix):
                        j_rows.update(
                            {r["id"]: r for r in read(RESULTS / "faithfulness" / judge / f"{dataset}_test.jsonl")}
                        )
                        v_rows.update(
                            {r["id"]: r for r in read(RESULTS / "faithfulness" / verifier / f"{dataset}_test.jsonl")}
                        )
                ids = sorted(set(j_rows) & set(v_rows))
                if not ids:
                    continue
                labels, v, _ = arrays([v_rows[i] for i in ids])
                j = [j_rows[i]["score"] for i in ids]
                j_lat = np.array([j_rows[i]["latency"] for i in ids])
                v_lat = np.array([v_rows[i]["latency"] for i in ids])
                judge_only = np.array([0.5 if s is None else s for s in j])
                base = [
                    judge,
                    verifier,
                    family,
                    len(ids),
                    fmt(metrics.auroc(labels, v)),
                    fmt(metrics.auroc(labels, judge_only)),
                ]
                for low, high in BANDS:
                    escalate = (v >= low) & (v <= high)
                    final = np.array([j[k] if escalate[k] and j[k] is not None else v[k] for k in range(len(ids))])
                    latency = float(np.mean(v_lat + escalate * j_lat))
                    rows.append(
                        base
                        + [
                            f"{low}-{high}",
                            f"{escalate.mean():.1%}",
                            fmt(metrics.auroc(labels, final)),
                            fmt(latency, 2),
                        ]
                    )
    header = [
        "judge",
        "verifier",
        "data",
        "n",
        "verifier AUROC",
        "judge AUROC",
        "band",
        "escalated",
        "cascade AUROC",
        "mean s",
    ]
    return "## Judge cascade (test examples scored by both)\n\n" + table(header, rows)


def retrieval_section():
    test, dev = read(RESULTS / "retrieval" / "test.jsonl"), read(RESULTS / "retrieval" / "dev.jsonl")
    if not test:
        return "## Retrieval\n\nNo results."
    platt_path = RESULTS / "retrieval" / "platt.json"
    platt = json.loads(platt_path.read_text()) if platt_path.exists() else None

    def scorers(rows):
        out = {"reranker (sigmoid)": [apply_platt(r["reranker"], 1.0, 0.0) for r in rows]}
        if platt:
            out["reranker (Platt)"] = [apply_platt(r["reranker"], platt["a"], platt["b"]) for r in rows]
        for name in rows[0]["cosine"]:
            out[f"cosine {name}"] = [np.asarray(r["cosine"][name]) for r in rows]
        return out

    original = [r for r in test if r["kind"] == "original"]
    chunk_labels = np.concatenate([r["relevant"] for r in original])
    query_labels = np.array([int(any(r["relevant"])) for r in test])
    dev_original = [r for r in dev if r["kind"] == "original"]
    dev_scores = scorers(dev_original) if dev_original else {}
    dev_labels = np.concatenate([r["relevant"] for r in dev_original]) if dev_original else None
    test_chunk = scorers(original)
    test_query = scorers(test)
    rows = []
    for name, per_query in test_chunk.items():
        chunk_scores = np.concatenate(per_query)
        query_scores = np.array([s.max() for s in test_query[name]])
        threshold = fmt(metrics.tuned_threshold(dev_labels, np.concatenate(dev_scores[name]))) if dev_scores else "—"
        calibration = fmt(metrics.ece(chunk_labels, chunk_scores)) if name.startswith("reranker") else "—"
        rows.append(
            [
                name,
                fmt(metrics.auroc(chunk_labels, chunk_scores)),
                fmt(metrics.auroc(query_labels, query_scores)),
                threshold,
                calibration,
            ]
        )
    p50, p95 = metrics.percentiles([r["latency"] for r in test])
    header = ["scorer", "chunk AUROC", "query AUROC", "best dev threshold", "ECE"]
    counts = (
        f"{len(chunk_labels)} chunks ({chunk_labels.mean():.1%} relevant); "
        f"{len(test)} queries ({query_labels.mean():.1%} with a relevant chunk). "
        f"Reranker latency p50 {p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms."
    )
    return "## Retrieval quality (RAGBench)\n\n" + counts + "\n\n" + table(header, rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def pareto_figure(summary):
    import matplotlib.pyplot as plt

    complete = {m: s for m, s in summary.items() if s["complete"]}
    if not complete:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style(ax)
    for method, s in complete.items():
        ax.scatter(s["p95"], s["auroc"], s=64, color=COLORS[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(method, (s["p95"], s["auroc"]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("mean p95 latency per answer (s)")
    ax.set_ylabel("macro AUROC (test)")
    ax.set_title("Accuracy vs latency", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=160)
    plt.close(fig)


def risk_coverage_figure(summary):
    import matplotlib.pyplot as plt

    ranked = sorted((m for m in summary if summary[m]["complete"]), key=lambda m: -summary[m]["auroc"])[: len(COLORS)]
    if not ranked:
        return
    rows = {m: {} for m in ranked}
    for m in ranked:
        for dataset in data.FAITHFULNESS_DATASETS:
            rows[m].update({r["id"]: r for r in read(RESULTS / "faithfulness" / m / f"{dataset}_test.jsonl")})
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style(ax)
    for color, m in zip(COLORS, ranked, strict=False):
        labels, scores, _ = arrays([rows[m][i] for i in ids])
        coverage, risk, aurc = metrics.risk_coverage(labels, scores)
        ax.plot(coverage, risk, color=color, linewidth=2, label=f"{m} (AURC {aurc:.3f})")
    ax.set_xlabel("coverage (fraction of answers shown)")
    ax.set_ylabel("unsupported rate among shown answers")
    ax.set_title(f"Risk vs coverage, {len(ids)} pooled test answers", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    faithfulness, summary = faithfulness_section()
    sections = [faithfulness, aggregation_section(), aggrefact_section(), cascade_section(), retrieval_section()]
    (OUT / "results.md").write_text("# CertainRAG benchmark results\n\n" + "\n\n".join(sections) + "\n")
    pareto_figure(summary)
    risk_coverage_figure(summary)
    print(f"wrote {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
