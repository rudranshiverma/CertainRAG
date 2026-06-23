import time
import random
import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from certainrag import CertainRAG

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class SimpleRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def index_documents(self, chunks: list[str]):
        self.chunks = [c.strip() for c in chunks if c and c.strip()]
        if not self.chunks:
            self.index = None
            return
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 4):
        if not self.chunks or self.index is None:
            return [], []
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        chunks_list, sims = [], []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunks_list.append(self.chunks[idx])
                sims.append(float(max(0.0, min(1.0, score))))
        return chunks_list, sims

def semantic_match(gold: str, answer: str, threshold: float = 0.65, return_score: bool = False):
    if not gold or not answer:
        return (0.0, False) if return_score else False
    emb1 = SIMILARITY_MODEL.encode(gold, convert_to_numpy=True)
    emb2 = SIMILARITY_MODEL.encode(answer, convert_to_numpy=True)
    score = util.cos_sim(emb1, emb2).item()
    is_correct = score >= threshold
    return (score, is_correct) if return_score else is_correct


def is_correct_pubmedqa(gold: str, answer: str) -> bool:
    """Robust categorical matching using word boundaries."""
    gold_clean = gold.strip().lower()
    ans_clean = answer.strip().lower()
    
    if gold_clean in ["yes", "no", "maybe"]:
        # FIXED: Using regex with word boundaries
        yes_count = len(re.findall(r'\byes\b', ans_clean))
        no_count = len(re.findall(r'\bno\b', ans_clean))
        maybe_count = len(re.findall(r'\b(maybe|uncertain|insufficient|unknown|not enough|insufficient evidence)\b', ans_clean))
        
        if yes_count > no_count and yes_count > maybe_count:
            return gold_clean == "yes"
        if no_count > yes_count and no_count > maybe_count:
            return gold_clean == "no"
        if maybe_count > 0:
            return gold_clean == "maybe"
        
        # Fallback
        return semantic_match(gold, answer, threshold=0.6)
    
    return semantic_match(gold, answer)

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def chunk_text(text: str, chunk_size: int = 550, overlap: int = 120) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    
    sentences = split_into_sentences(text)
    chunks = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += (" " + sent) if current else sent
    
    if current:
        chunks.append(current.strip())
    
    return [c for c in chunks if len(c.strip()) > 50] or [text]

def load_pubmedqa(n: int = 120):
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    subset = dataset.select(range(min(n, len(dataset))))
    queries, contexts, answers = [], [], []
    for item in subset:
        queries.append(item["question"])
        ctx = item.get("context", {})
        context_list = ctx.get("contexts", []) if isinstance(ctx, dict) else [str(ctx)]
        contexts.append(context_list)
        answers.append(str(item.get("final_decision", "")))
    return queries, contexts, answers


def load_squad(n: int = 120):
    dataset = load_dataset("rajpurkar/squad", split="validation")
    n = min(n, len(dataset))
    subset = dataset.select(range(n))

    queries, contexts, answers = [], [], []
    for item in subset:
        queries.append(item["question"])
        contexts.append([item["context"]])  # single-element list, unchunked
        answers.append(item["answers"]["text"][0] if item["answers"]["text"] else "")

    return queries, contexts, answers

def run_single_query(rag, retriever, llm, llm_varied, query, gold_answer, domain, compute_entropy=True, n_entropy_samples=5):
    t0 = time.time()
    chunks, retrieval_scores = retriever.retrieve(query, top_k=4)
    if not chunks:
        return None

    context_str = "\n\n".join(chunks)
    prompt = f"""Answer using only the provided context. Be concise and direct.

Context:
{context_str}

Question: {query}
Answer:"""

    try:
        answer = llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f" [generation failed] {e}")
        return None

    responses = None
    if compute_entropy:
        try:
            responses = [llm_varied.invoke(prompt).content.strip() for _ in range(n_entropy_samples)]
        except Exception as e:
            print(f" [entropy sampling failed] {e}")
            responses = None

    try:
        result = rag.evaluate(answer=answer, context_chunks=chunks, retrieval_scores=retrieval_scores, responses=responses)
    except Exception as e:
        print(f" [CertainRAG evaluation failed] {e}")
        return None

    if domain == "medical":
        is_correct = is_correct_pubmedqa(gold_answer, answer)
    else:
        is_correct = semantic_match(gold_answer, answer)

    total_latency = (time.time() - t0) * 1000

    return {
        "domain": domain,
        "query": query[:250],
        "answer": answer[:600],
        "gold": gold_answer,
        "is_correct": is_correct,
        "uncertainty_score": result.uncertainty_score,
        "uncertainty_level": result.uncertainty_level,
        "should_abstain": result.should_abstain,
        "retrieval_confidence": result.retrieval_confidence,
        "faithfulness_score": result.faithfulness_score,
        "semantic_entropy": result.semantic_entropy,
        "n_supporting_chunks": len(result.supporting_chunks),
        "n_contradicting_chunks": len(result.contradicting_chunks),
        "explanation": result.explanation,
        "total_latency_ms": round(total_latency, 2),
        "certainrag_latency_ms": result.latency_ms
    }


def run_benchmark(domain, queries, contexts, gold_answers, n=120, compute_entropy=True):
    print(f"\n=== Running benchmark: {domain} ===")
    rag = CertainRAG(fast_mode=True)
    retriever = SimpleRetriever()
    llm = ChatOllama(model="mistral", temperature=0.0)
    llm_varied = ChatOllama(model="mistral", temperature=0.7)

    results = []
    for i in range(min(n, len(queries))):
        retriever.index_documents(contexts[i])
        row = run_single_query(rag, retriever, llm, llm_varied, queries[i], gold_answers[i], domain, compute_entropy)
        if row:
            results.append(row)
        if (i + 1) % 20 == 0:
            print(f" {domain}: {i+1}/{min(n, len(queries))} done")
    print(f"=== {domain} complete: {len(results)} successful ===")
    return results


def main():
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    all_results = []

    print("Loading PubMedQA...")
    try:
        q, c, a = load_pubmedqa(n=20)   # reduced for faster testing
        all_results.extend(run_benchmark("medical", q, c, a, n=100))
    except Exception as e:
        print(f"PubMedQA failed: {e}")

    print("\nLoading SQuAD...")
    try:
        q, c, a = load_squad(n=100)
        all_results.extend(run_benchmark("squad", q, c, a, n=100))
    except Exception as e:
        print(f"Squad failed: {e}")

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    print(f"\n✅ Saved {len(df)} results to evaluation/benchmark_results.csv")


if __name__ == "__main__":
    main()