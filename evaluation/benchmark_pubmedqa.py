import hashlib
import json
import os
import random
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from certainrag import CertainRAG, utils, OllamaClient
from .retriever import SimpleRetriever
from certainrag.scorer import combine_scores

OUTPUT_DIR=os.path.join(os.path.dirname(__file__), "outputs_pubmed")
RANDOM_SEED=42
DECISION_PATTERN=re.compile(r"^\s*(yes|no|maybe)\b", re.IGNORECASE)

def build_dataset(n_examples=150, n_distractors=60, seed=RANDOM_SEED):
    from datasets import load_dataset
    rng=random.Random(seed)
    dataset=load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    indices=list(range(len(dataset)))
    rng.shuffle(indices)

    #a shared pool of "wrong topic" distractor sentences for retrieval to ignore
    distractor_pool = []
    for idx in indices[:n_distractors]:
        row = dataset[idx]
        context_text = " ".join(row["context"]["contexts"])
        distractor_pool.extend(utils.split_into_sentences(context_text))
    rng.shuffle(distractor_pool)

    examples = []
    for idx in indices[n_distractors:]:
        if len(examples) >= n_examples:
            break
        row=dataset[idx]
        context_text = " ".join(row["context"]["contexts"])
        own_sentences = utils.split_into_sentences(context_text)
        if not own_sentences:
            continue

        distractors = rng.sample(distractor_pool, min(20, len(distractor_pool)))
        chunk_pool = []
        for sentence in own_sentences + distractors:
            if sentence not in chunk_pool:
                chunk_pool.append(sentence)

        examples.append({
            "question": row["question"],
            "own_sentences": set(own_sentences),
            "chunk_pool": chunk_pool,
            "gold_decision": row["final_decision"].strip().lower(),
        })

    return examples


def extract_decision(answer_text):
    match = DECISION_PATTERN.search(answer_text)
    if match:
        return match.group(1).lower()
    return None


def build_answer_prompt(question, chunks):
    context = "\n".join(chunks)
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer the question directly using only the context above. "
        "Start with Yes, No, or Maybe, then briefly explain."
    )


def build_corruption_prompt(question, chunks, real_answer):
    context = "\n".join(chunks)
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        f"Here is a correct answer: {real_answer}\n\n"
        "Rewrite this answer so it contains exactly one subtle factual error "
        "(a flipped negation, a changed number, or an unsupported claim not in "
        "the context) while still sounding plausible and confident. Do not "
        "mention that it is wrong. Output only the rewritten answer."
    )


def question_id(question):
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def load_checkpoint(checkpoint_path):
    rows = []
    done_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)
                done_ids.add(row["question_id"])
    return rows, done_ids


def run_pipeline(
    examples,
    generator_client,
    judge_client,
    embedder_name=utils.DEFAULT_EMBEDDER_MODEL,
    top_k=4,
    checkpoint_path=None,
    use_self_consistency=False,
):
    embedder = utils.get_embedder(embedder_name)
    rag = CertainRAG(
        llm_client=generator_client,
        judge_llm_client=judge_client,
        use_self_consistency=use_self_consistency,
        n_samples=3,
    )

    rows, done_ids = load_checkpoint(checkpoint_path)
    if done_ids:
        print(f"Resuming from checkpoint: {len(done_ids)} rows already done.")

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

                hit = any(c in ex["own_sentences"] for c in chunks)
                rank = None
                for j, c in enumerate(chunks):
                    if c in ex["own_sentences"]:
                        rank = j + 1
                        break

                answer_prompt = build_answer_prompt(ex["question"], chunks)
                real_answer= generator_client.generate(answer_prompt,temperature=0.0, n=1)[0]

                corruption_prompt = build_corruption_prompt(ex["question"], chunks, real_answer)
                unfaithful_answer = generator_client.generate(corruption_prompt, temperature=0.7, n=1)[0]

                real_result = rag.score(
                    ex["question"], real_answer, chunks,
                    retrieval_scores=scores)
                unfaithful_result = rag.score(
                    ex["question"], unfaithful_answer, chunks,
                    retrieval_scores=scores,
                )

                predicted_decision = extract_decision(real_answer)
                is_wrong = predicted_decision is not None and predicted_decision != ex["gold_decision"]

                row = {
                    "question_id": qid,
                    "question": ex["question"],
                    "real_answer": real_answer,
                    "unfaithful_answer": unfaithful_answer,
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
                print(f"[{i}] skipped due to error: {e}")
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
    labels = np.array(labels)
    scores = np.array(scores)
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.linspace(0, 1, 101):
        predictions = (scores >= t).astype(int)
        accuracy = (predictions == labels).mean()
        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = t
    return best_acc, best_thresh


def compute_faithfulness_metrics(rows):
    labels = []
    scores = []
    for r in rows:
        if r["faithful_score"] is None or r["unfaithful_score"] is None:
            continue
        labels.append(1)
        scores.append(r["faithful_score"])
        labels.append(0)
        scores.append(r["unfaithful_score"])

    if len(set(labels)) < 2:
        return {}, labels, scores

    auc = roc_auc_score(labels, scores)
    accuracy, threshold = best_threshold_accuracy(labels, scores)
    predictions_at_half = (np.array(scores) >= 0.5).astype(int)

    metrics = {
        "auc": auc,
        "best_threshold": threshold,
        "accuracy_at_best_threshold": accuracy,
        "accuracy_at_0.5": float((predictions_at_half == np.array(labels)).mean()),
        "n_pairs": len(labels) // 2,
    }
    return metrics, labels, scores


def compute_retrieval_metrics(rows):
    hits = [1 if r["retrieval_hit"] else 0 for r in rows]
    reciprocal_ranks = [1.0/r["retrieval_rank"] if r["retrieval_rank"] else 0.0 for r in rows]
    return {
        "hit_rate_at_k": sum(hits)/len(hits) if hits else None,
        "mrr": sum(reciprocal_ranks)/len(reciprocal_ranks) if reciprocal_ranks else None,
        "n_examples": len(rows),
    }

def compute_uncertainty_metrics(rows):
    labels = []
    scores = []
    for r in rows:
        if r["uncertainty"] is None:
            continue
        labels.append(1 if r["is_wrong"] else 0)
        scores.append(r["uncertainty"])

    if len(set(labels)) < 2:
        return {"auc": None, "n_examples": len(scores)}

    auc = roc_auc_score(labels, scores)
    return {"auc": auc, "n_examples": len(scores)}, labels,scores


def compute_decision_accuracy(rows):
    correct = sum(1 for r in rows if not r["is_wrong"] and r["real_answer"])
    total = sum(1 for r in rows if r["real_answer"])
    return {
        "decision_accuracy": float(correct / total) if total else None,
        "n_examples": total,
    }
 
def plot_faithfulness_roc(labels, scores, auc, path):
    plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"LLM judge (AUC={auc:.3f})", color="tab:blue")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Faithfulness ROC: PubMedQA")
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
    plt.title("Faithfulness Score Distribution: PubMedQA")
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
    plt.title(f"Retrieval Rank Distribution: PubMedQA\n({misses} misses not shown)")
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
    plt.title("Uncertainty → Wrong Answer ROC: PubMedQA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(n_examples=150, top_k=4, generator_model="mistral", judge_model="llama3", use_self_consistency=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building PubMedQA dataset...")
    examples = build_dataset(n_examples=n_examples)
    print(f"Built {len(examples)} questions.")

    generator_client = OllamaClient(model=generator_model)
    judge_client = OllamaClient(model=judge_model)

    checkpoint_path = os.path.join(OUTPUT_DIR, "pipeline_checkpoint.jsonl")
    print(f"Running pipeline (generator={generator_model}, judge={judge_model})...")
    rows = run_pipeline(
        examples, generator_client, judge_client,
        top_k=top_k, checkpoint_path=checkpoint_path,
        use_self_consistency=use_self_consistency,
    )
    print(f"Completed {len(rows)} rows.")

    faithfulness_metrics, labels, scores = compute_faithfulness_metrics(rows)
    retrieval_metrics = compute_retrieval_metrics(rows)
    uncertainty_result = compute_uncertainty_metrics(rows)
    decision_metrics = compute_decision_accuracy(rows)
    uncertainty_metrics = uncertainty_result[0] if isinstance(uncertainty_result, tuple) else uncertainty_result
    u_labels = uncertainty_result[1] if isinstance(uncertainty_result, tuple) else []
    u_scores = uncertainty_result[2] if isinstance(uncertainty_result, tuple) else []

    if faithfulness_metrics:
        plot_faithfulness_roc(
            labels, scores, faithfulness_metrics["auc"],
            os.path.join(OUTPUT_DIR, "faithfulness_roc.png"),
        )
    plot_retrieval_ranks(rows, os.path.join(OUTPUT_DIR, "retrieval_ranks.png"))
    if u_labels:
        plot_uncertainty_roc(u_labels, u_scores, uncertainty_metrics["auc"],
            os.path.join(OUTPUT_DIR, "uncertainty_roc.png"))
    all_metrics = {
        "retrieval": retrieval_metrics,
        "faithfulness": faithfulness_metrics,
        "combined_uncertainty": uncertainty_metrics,
        "decision_accuracy": decision_metrics,
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(json.dumps(all_metrics, indent=2))
    return all_metrics


if __name__ == "__main__":
    main()