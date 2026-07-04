import hashlib
import json
import os
import random
import re
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from certainrag import CertainRAG, OllamaClient, utils
from .retriever import SimpleRetriever

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_squad")
RANDOM_SEED = 42
F1_THRESHOLD = 0.5


def sliding_window_chunks(sentences, window=2, overlap=1):
    if not sentences:
        return []
    chunks = []
    step = window - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + window])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def build_dataset(n_examples=150, n_distractors=60, seed=RANDOM_SEED):
    from datasets import load_dataset
    rng = random.Random(seed)
    dataset = load_dataset("rajpurkar/squad", split="validation")
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    distractor_pool = []
    for idx in indices[:n_distractors]:
        row = dataset[idx]
        sentences = utils.split_into_sentences(row["context"])
        distractor_pool.extend(sliding_window_chunks(sentences, window=2, overlap=1))
    rng.shuffle(distractor_pool)

    examples = []
    for idx in indices[n_distractors:]:
        if len(examples) >= n_examples:
            break
        row = dataset[idx]
        gold_answer = row["answers"]["text"][0] if row["answers"]["text"] else None
        if not gold_answer:
            continue
        sentences = utils.split_into_sentences(row["context"])
        own_chunks = sliding_window_chunks(sentences, window=2, overlap=1)
        if not own_chunks:
            continue
        gold_chunk = next((c for c in own_chunks if gold_answer.lower() in c.lower()), None)
        if gold_chunk is None:
            continue
        distractors = rng.sample(distractor_pool, min(20, len(distractor_pool)))
        chunk_pool = list(dict.fromkeys(own_chunks + distractors))
        examples.append({
            "question": row["question"],
            "gold_chunk": gold_chunk,
            "chunk_pool": chunk_pool,
            "gold_answer": gold_answer,
        })
    return examples


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(predicted, gold):
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {t: min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens) & set(gold_tokens)}
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_answer_prompt(question, chunks):
    context = "\n".join(chunks)
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer using only the context above. Give the shortest possible answer "
        "(a word, name, date, or number). No explanation."
    )


def build_corruption_prompt(question, chunks, real_answer):
    context = "\n".join(chunks)
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        f"Correct answer: {real_answer}\n\n"
        "Rewrite this answer with one subtle error: a wrong number, name, date, "
        "or entity not in the context. Keep it short and plausible. "
        "Output only the rewritten answer."
    )


def question_id(question):
    return hashlib.sha1(question.encode()).hexdigest()[:16]

def load_checkpoint(path):
    rows, done_ids = [], set()
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows.append(row)
                    done_ids.add(row["question_id"])
    return rows, done_ids

def run_pipeline(examples, generator_client, judge_client, top_k=4, checkpoint_path=None, use_self_consistency=False):
    embedder = utils.get_embedder()
    rag = CertainRAG(
        llm_client=generator_client,
        judge_llm_client=judge_client,
        use_self_consistency=use_self_consistency,
        n_samples=3,
    )
    rows, done_ids = load_checkpoint(checkpoint_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} rows already done.")
    checkpoint_file = open(checkpoint_path, "a") if checkpoint_path else None
    try:
        for i, ex in enumerate(examples):
            qid = question_id(ex["question"])
            if qid in done_ids:
                continue
            try:
                retriever = SimpleRetriever(embedder, ex["chunk_pool"])
                chunks, scores = retriever.retrieve(ex["question"], top_k=top_k)
                if not chunks:
                    continue
                hit = ex["gold_chunk"] in chunks
                rank = next((j + 1 for j, c in enumerate(chunks) if c == ex["gold_chunk"]), None)
                real_answer = generator_client.generate(build_answer_prompt(ex["question"], chunks), temperature=0.0, n=1)[0]
                unfaithful_answer = generator_client.generate(build_corruption_prompt(ex["question"], chunks, real_answer), temperature=0.7, n=1)[0]
                real_result = rag.score(ex["question"], real_answer, chunks, retrieval_scores=scores)
                unfaithful_result = rag.score(ex["question"], unfaithful_answer, chunks, retrieval_scores=scores)
                answer_f1 = float(f1_score(real_answer, ex["gold_answer"]))
                is_wrong = answer_f1 < F1_THRESHOLD
                row = {
                    "question_id": qid,
                    "question": ex["question"],
                    "gold_answer": ex["gold_answer"],
                    "real_answer": real_answer,
                    "unfaithful_answer": unfaithful_answer,
                    "answer_f1": answer_f1,
                    "retrieval_hit": hit,
                    "retrieval_rank": rank,
                    "retrieval_confidence": real_result.retrieval_confidence,
                    "faithful_score": real_result.faithfulness,
                    "unfaithful_score": unfaithful_result.faithfulness,
                    "self_consistency": real_result.self_consistency,
                    "is_wrong": is_wrong,
                    "uncertainty": real_result.uncertainty,
                }
            except Exception as e:
                print(f"[{i}] skipped: {e}")
                continue
            rows.append(row)
            done_ids.add(qid)
            if checkpoint_file:
                checkpoint_file.write(json.dumps(row) + "\n")
                checkpoint_file.flush()
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(examples)} ({len(rows)} successful)")
    finally:
        if checkpoint_file:
            checkpoint_file.close()
    return rows

def best_threshold_accuracy(labels, scores):
    labels, scores = np.array(labels), np.array(scores)
    best_acc, best_thresh = 0.0, 0.5
    for t in np.linspace(0, 1, 101):
        acc = ((scores >= t).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return float(best_acc), float(best_thresh)

def compute_faithfulness_metrics(rows):
    labels, scores = [], []
    for r in rows:
        if r["faithful_score"] is None or r["unfaithful_score"] is None:
            continue
        labels += [1, 0]
        scores += [r["faithful_score"], r["unfaithful_score"]]
    if len(set(labels)) < 2:
        return {}, [], []
    auc = float(roc_auc_score(labels, scores))
    acc, thresh = best_threshold_accuracy(labels, scores)
    preds = (np.array(scores) >= 0.5).astype(int)
    return {
        "auc": auc,
        "best_threshold": thresh,
        "accuracy_at_best_threshold": acc,
        "accuracy_at_0.5": float((preds == np.array(labels)).mean()),
        "n_pairs": len(labels) // 2,
    }, labels, scores

def compute_retrieval_metrics(rows):
    hits = [1 if r["retrieval_hit"] else 0 for r in rows]
    rr = [1.0 / r["retrieval_rank"] if r["retrieval_rank"] else 0.0 for r in rows]
    return {
        "hit_rate_at_k": float(sum(hits) / len(hits)) if hits else None,
        "mrr": float(sum(rr) / len(rr)) if rr else None,
        "n_examples": len(rows),
    }

def compute_uncertainty_metrics(rows):
    labels = [1 if r["is_wrong"] else 0 for r in rows if r["uncertainty"] is not None]
    scores = [r["uncertainty"] for r in rows if r["uncertainty"] is not None]
    if len(set(labels)) < 2:
        return {"auc": None, "n_examples": len(scores)}, [], []
    return {"auc": float(roc_auc_score(labels, scores)), "n_examples": len(scores)}, labels, scores

def compute_answer_metrics(rows):
    f1s = [r["answer_f1"] for r in rows]
    correct = sum(1 for f in f1s if f >= F1_THRESHOLD)
    return {
        "mean_f1": float(sum(f1s) / len(f1s)) if f1s else None,
        "accuracy_at_f1_0.5": float(correct / len(f1s)) if f1s else None,
        "n_examples": len(rows),
    }

def plot_faithfulness_roc(labels, scores, auc, path):
    plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"LLM judge (AUC={auc:.3f})", color="tab:blue")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Faithfulness ROC — SQuAD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_score_distributions(rows, path):
    faithful = [r["faithful_score"] for r in rows if r["faithful_score"] is not None]
    unfaithful = [r["unfaithful_score"] for r in rows if r["unfaithful_score"] is not None]
    plt.figure(figsize=(7, 5))
    plt.hist(faithful, bins=20, alpha=0.6, label="faithful", color="tab:green")
    plt.hist(unfaithful, bins=20, alpha=0.6, label="unfaithful", color="tab:red")
    plt.xlabel("Faithfulness score")
    plt.ylabel("Count")
    plt.title("Faithfulness Score Distribution — SQuAD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_retrieval_ranks(rows, path):
    ranks = [r["retrieval_rank"] for r in rows if r["retrieval_rank"] is not None]
    misses = sum(1 for r in rows if not r["retrieval_hit"])
    plt.figure(figsize=(6, 5))
    plt.hist(ranks, bins=range(1, 7), align="left", rwidth=0.7, color="tab:blue")
    plt.xlabel("Rank of gold chunk")
    plt.ylabel("Count")
    plt.title(f"Retrieval Rank Distribution — SQuAD\n({misses} misses not shown)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_f1_distribution(rows, path):
    f1s = [r["answer_f1"] for r in rows]
    plt.figure(figsize=(6, 5))
    plt.hist(f1s, bins=20, color="tab:purple", alpha=0.8)
    plt.axvline(F1_THRESHOLD, color="red", linestyle="--", label=f"Threshold={F1_THRESHOLD}")
    plt.xlabel("Answer F1 score")
    plt.ylabel("Count")
    plt.title("Answer F1 Distribution — SQuAD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_uncertainty_roc(labels, scores, auc, path):
    plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"Combined uncertainty (AUC={auc:.3f})", color="tab:orange")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Uncertainty → Wrong Answer ROC — SQuAD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main(n_examples=150, top_k=4, generator_model="mistral", judge_model="llama3", use_self_consistency=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building SQuAD dataset...")
    examples = build_dataset(n_examples=n_examples)
    print(f"Built {len(examples)} questions.")
    generator_client = OllamaClient(model=generator_model)
    judge_client = OllamaClient(model=judge_model)
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.jsonl")
    print(f"Running pipeline (generator={generator_model}, judge={judge_model})...")
    rows = run_pipeline(examples, generator_client, judge_client, top_k=top_k, checkpoint_path=checkpoint_path, use_self_consistency=use_self_consistency)
    print(f"Completed {len(rows)} rows.")

    faithfulness_metrics, f_labels, f_scores = compute_faithfulness_metrics(rows)
    retrieval_metrics = compute_retrieval_metrics(rows)
    uncertainty_metrics, u_labels, u_scores = compute_uncertainty_metrics(rows)
    answer_metrics = compute_answer_metrics(rows)

    if f_labels:
        plot_faithfulness_roc(f_labels, f_scores, faithfulness_metrics["auc"], os.path.join(OUTPUT_DIR, "faithfulness_roc.png"))
        plot_score_distributions(rows, os.path.join(OUTPUT_DIR, "score_distributions.png"))
    plot_retrieval_ranks(rows, os.path.join(OUTPUT_DIR, "retrieval_ranks.png"))
    plot_f1_distribution(rows, os.path.join(OUTPUT_DIR, "f1_distribution.png"))
    if u_labels:
        plot_uncertainty_roc(u_labels, u_scores, uncertainty_metrics["auc"], os.path.join(OUTPUT_DIR, "uncertainty_roc.png"))

    all_metrics = {
        "retrieval": retrieval_metrics,
        "faithfulness": faithfulness_metrics,
        "combined_uncertainty": uncertainty_metrics,
        "answer_accuracy": answer_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(json.dumps(all_metrics, indent=2))
    return all_metrics

if __name__ == "__main__":
    main()