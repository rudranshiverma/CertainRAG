# CertainRAG

**Runtime uncertainty quantification for Retrieval-Augmented Generation pipelines.**

Most RAG systems generate an answer and show it. CertainRAG sits between generation and the user, scoring how much you should trust that answer before deciding whether to show it, regenerate it, or abstain.



## The Problem

RAG pipelines fail in two distinct ways that are invisible to the user:

1. **Retrieval failure**: the retriever returns irrelevant or contradictory chunks, and the model generates a confident-sounding answer from bad evidence.
2. **Faithfulness failure**: the retriever finds the right chunks, but the model fabricates, misquotes, or contradicts them in its answer.

Neither failure is detectable from the answer text alone. CertainRAG quantifies both at runtime, giving your application a signal to act on before the user sees anything.



## How It Works

CertainRAG computes three independent signals from what your RAG pipeline already produces — no extra infrastructure required.

### Signal 1: Retrieval Confidence

Takes the similarity scores your retriever already computed and aggregates them into a single confidence value using rank-weighted averaging (higher-ranked chunks weighted more, using logarithmic decay). A score near 0 means the retriever was scraping the bottom of the barrel; near 1 means strong, relevant evidence was found.

**Input:** cosine similarity scores from your retriever, in any order  
**Output:** float in [0, 1]  
**Cost:** zero — pure arithmetic over scores you already have

> **API contract:** scores must be cosine similarities in [0, 1]. Standard for FAISS, Pinecone, ChromaDB (cosine mode), and Weaviate cosine retrievers. BM25 scores require caller-side conversion.

### Signal 2: Faithfulness

Uses a separate judge LLM to evaluate whether the generated answer is grounded in the retrieved context. The judge is prompted to explicitly state the answer's core claim, verify it against the context, and score with calibrated anchors that penalize negation flips, fabricated facts, and contradictions — not just topical relevance.

Using a **different model as judge than as generator** is intentional: it mitigates self-preference bias, where a model tends to rate its own outputs highly regardless of faithfulness.

**Input:** question, generated answer, retrieved chunks  
**Output:** float in [0, 1] + one-sentence reasoning string  
**Cost:** one LLM call per query (the judge)

### Signal 3 (Optional): Self-Consistency

Samples the generator model multiple times for the same question at higher temperature, then measures how much the answers diverge in embedding space. High dispersion means the model is genuinely uncertain — it keeps landing on different meanings. Low dispersion means it's confident regardless of phrasing variation.

**Input:** question, generator LLM  
**Output:** float in [0, 1] (0 = consistent/confident, 1 = highly dispersed)  
**Cost:** n_samples extra LLM calls per query (default: 3)

### Combined Uncertainty Score

The three signals are combined via weighted average into a single `uncertainty` score in [0, 1]. Default weights: faithfulness 0.5, retrieval confidence 0.3, self-consistency 0.2 (renormalized when self-consistency is disabled). Weights are customizable and can be calibrated to your domain using the included `evaluation/calibrate_weights.py` script.

```
uncertainty = 1 - weighted_average(retrieval_confidence, faithfulness, 1 - self_consistency)
```

A high uncertainty score does not mean the answer is wrong — it means the evidence trail is weak or the answer is poorly grounded, and your application should decide whether to regenerate, abstain, or flag for human review.



## Runtime Flow

```
User Query
    │
    ▼
Your Retriever ──────────────────────────────► chunks + similarity scores
    │                                                       │
    ▼                                                       │
Your LLM Generator ──────────────────────────► generated answer
                                                            │
                                          ┌─────────────────┘
                                          ▼
                                    CertainRAG.score()
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                   Retrieval         Faithfulness    Self-Consistency
                   Confidence         (judge LLM)    (optional, generator)
                          │               │               │
                          └───────────────┴───────────────┘
                                          │
                                          ▼
                                  uncertainty score
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                        Show          Regenerate       Abstain
```



## Installation

```bash
pip install certainrag
```

**Dependencies:** `requests` (only if using OllamaClient), `sentence-transformers` (only if using self-consistency)

CertainRAG works with any LLM backend via the `LLMClient` interface — OpenAI, Anthropic, Gemini, vLLM, or any hosted API. `OllamaClient` is the bundled implementation for fully local, offline inference:

```bash
ollama pull mistral   # generator
ollama pull llama3    # judge 
```



## Usage

### Bring your own LLM client

CertainRAG works with any LLM backend. Subclass `LLMClient` and implement one method:

```python
from certainrag import LLMClient, CertainRAG

class OpenAIClient(LLMClient):
    def generate(self, prompt, temperature=0.0, n=1):
        import openai
        responses = []
        for _ in range(n):
            r = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            responses.append(r.choices[0].message.content.strip())
        return responses

rag = CertainRAG(
    llm_client=OpenAIClient(),
    judge_llm_client=OpenAIClient(),  # can be a different model/instance
)
```

`OllamaClient` is the bundled implementation for fully local inference:

```python
from certainrag import CertainRAG, OllamaClient

rag = CertainRAG(
    llm_client=OllamaClient(model="mistral"),
    judge_llm_client=OllamaClient(model="llama3"),
)
```

### Basic — two core signals



```python
rag = CertainRAG(
    llm_client=OllamaClient(model="mistral"),
    judge_llm_client=OllamaClient(model="llama3"),  # required for faithfulness
)

result = rag.score(
    question=question,
    answer=answer,
    chunks=chunks,
    retrieval_scores=scores,  # required: cosine similarities from your retriever
)

if result.uncertainty > 0.6:
    # regenerate or abstain
    pass
```

### With self-consistency

```python
rag = CertainRAG(
    llm_client=OllamaClient(model="mistral"),
    judge_llm_client=OllamaClient(model="llama3"),
    use_self_consistency=True,
    n_samples=3,
)
```

### Custom weights

```python
rag = CertainRAG(
    llm_client=OllamaClient(model="mistral"),
    judge_llm_client=OllamaClient(model="llama3"),
    weights={
        "retrieval_confidence": 0.2,
        "faithfulness": 0.6,
        "self_consistency": 0.2,
    },
)
```

### Result object

```python
result.uncertainty          # float [0, 1] — combined uncertainty score
result.retrieval_confidence # float [0, 1] — how relevant was the retrieved evidence
result.faithfulness         # float [0, 1] — how grounded is the answer in the chunks
result.faithfulness_reasoning  # str — one-sentence judge reasoning
result.self_consistency     # float [0, 1] or None if disabled
```



## Benchmark Results

Evaluated on **SQuAD** (general-domain QA, 1000 examples) and **PubMedQA** (biomedical yes/no/maybe reasoning, 488 examples). Generator: Mistral 7B. Judge: LLaMA 3 8B (separate model to avoid self-preference bias). Retriever: FAISS with sentence-transformers/all-MiniLM-L6-v2.

Faithfulness is evaluated using a **contrastive paired setup**: for each question, a real answer and a subtly corrupted answer (negation flip, fabricated entity, or unsupported claim) are scored against the same retrieved chunks. AUC measures how well the signal separates faithful from unfaithful answers.

| Metric                                 | SQuAD (n=1000) | PubMedQA (n=488) |
|----------------------------------------|----------------|------------------|
| Retrieval hit-rate @ k=4               | 0.964          | 1.000            |
| Retrieval MRR                          | 0.758          | 0.996            |
| Faithfulness AUC                       | 0.816          | 0.821            |
| Faithfulness accuracy @ best threshold | 0.786          | 0.787            |
| Combined uncertainty AUC               |                |                  |



### What these numbers mean

**Faithfulness AUC ~0.82 across both domains** is the core result — the judge signal consistently separates grounded from corrupted answers regardless of domain (general vs biomedical). Consistency across domains is more meaningful than a high number on one dataset.

**Combined uncertainty AUC against gold correctness labels is deliberately not the headline metric.** CertainRAG detects whether an answer is grounded and internally consistent, not whether it is factually correct. These are different constructs: a model can produce a perfectly faithful answer that still disagrees with the gold label due to reasoning errors, and the converse is also true. Reporting combined uncertainty AUC as a correctness predictor would be overclaiming; the faithfulness AUC is the appropriate signal-quality metric.



## Project Structure

```
certainrag/
├── certainrag/
│   ├── __init__.py          # public API: CertainRAG class
│   ├── scorer.py            # CertainRAGResult + combine_scores()
│   ├── llm_client.py        # LLMClient ABC + OllamaClient
│   ├── utils.py             # shared utilities
│   ├── exceptions.py        # CertainRAGError hierarchy
│   └── signals/
│       ├── retrieval_score.py   # RetrievalConfidenceSignal
│       ├── faithfulness.py      # FaithfulnessSignal (LLM-judge)
│       └── self_consistency.py  # SelfConsistencySignal (optional)
└── evaluation/
    ├── benchmark.py             # PubMedQA end-to-end benchmark
    ├── benchmark_squad.py       # SQuAD end-to-end benchmark
    ├── retriever.py             # FAISS-based retriever for benchmarks
    ├── llm_client.py            # re-exports from certainrag.llm_client
    └── calibrate_weights.py     # fit signal weights from checkpoint data
```



## Running the Benchmarks

```bash
# PubMedQA
python -m evaluation.benchmark

# SQuAD
python -m evaluation.benchmark_squad

# With self-consistency
python -m evaluation.benchmark_squad  # set use_self_consistency=True in main()

# Calibrate weights from a completed checkpoint
python -m evaluation.calibrate_weights evaluation/outputs_squad/checkpoint.jsonl
```

Both benchmarks checkpoint after every row and resume automatically on restart. Outputs saved to `evaluation/outputs_pubmed/` and `evaluation/outputs_squad/`.



## Design Decisions

A log of non-obvious architectural choices is maintained in `DECISIONS.md`. Key entries:

**Why LLM-as-judge over NLI cross-encoder for faithfulness:** NLI models (e.g. DeBERTa-v3) operate at the textual entailment level — they score whether a hypothesis is entailed by a premise as a literal textual relationship. Generated answers are multi-clause, judgment-prefixed ("Yes, because..."), and often inferential rather than verbatim. These properties systematically collapse NLI entailment scores toward near-zero regardless of actual faithfulness. Isolated diagnostic testing confirmed this: a clean atomic claim ("vaginal pH can be measured from the wet mount slide") scored 0.9975 entailment, while the same fact embedded in a generated sentence with causal reasoning scored 0.001. LLM-as-judge handles inference, paraphrase, and synthesis correctly.

**Why a separate judge model:** using the same model to generate and judge its own output introduces self-preference bias — models consistently rate their own outputs more favorably. Separating generator and judge is standard practice in LLM evaluation research.

**Why retrieval scores are not normalized:** min-max normalization within a single retrieval call destroys absolute score information. A retrieval returning [0.02, 0.02, 0.02] (nothing relevant found) would normalize to 1.0 (maximum confidence) — the opposite of the intended signal. Absolute cosine similarities are preserved so low retrieval quality propagates as genuine low confidence into the combined score.

**Why self-consistency uses embedding dispersion rather than NLI clustering:** semantic entropy via NLI clustering (Kuhn et al. 2023) requires O(n²) pairwise NLI calls per query — prohibitively expensive at runtime with n_samples=3-5. Embedding dispersion is a cheaper proxy that correlates with semantic variance without the quadratic cost. The tradeoff is lower precision in distinguishing meaning clusters, accepted in exchange for practical runtime cost.



## Limitations

- **Faithfulness ≠ correctness.** CertainRAG detects whether an answer is grounded in retrieved context, not whether the retrieved context or the answer is factually true. A grounded wrong answer scores low uncertainty; an ungrounded right answer scores high uncertainty.
- **Paraphrase corruptions.** When a corrupted answer is a semantic paraphrase of the original (synonym substitution, word reordering), the LLM judge may not distinguish it from the faithful version. The signal works best on factual flips, negations, and fabricated specifics.
- **Retrieval scores must be cosine similarities.** The retrieval confidence signal assumes [0, 1] bounded cosine similarities. BM25 or distance-based scores require caller-side conversion.
- **Self-consistency adds latency.** With n_samples=3, self-consistency adds 3 extra LLM calls per query (~6-15 seconds on a local 7B model). Not recommended as a default for latency-sensitive applications.

