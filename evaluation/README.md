# Evaluation

Benchmarks for the two CertainRAG signals. Every run writes raw scores to `evaluation/results/`, and every metric, threshold and figure is computed from those files by `report.py`, so changing an analysis never requires re-running a model.

## Faithfulness

Each method scores whether an answer is supported by its context. Scores are support probabilities (higher means more faithful), and labels are 1 for supported answers.

| Dataset | Unit | Test | Threshold tuning | Label source |
|---|---|---|---|---|
| RAGTruth QA, Summary | response | full test split (900 each) | 100 train sources per task (600 responses) | human span annotation; any span means unsupported |
| RAGBench: PubMedQA, CovidQA, TechQA, eManual, CUAD, DelucionQA | response | 400 sampled per subset | 150 sampled from the validation split | `adherence_score` (GPT-4 annotation) |
| LLM-AggreFact | claim | full test split | 300 sampled per constituent dataset from `dev` | human |

RAGTruth Data2txt is excluded because its sources are JSON records rather than prose. The RAGTruth dev split is built from training sources only, so no source document appears in both dev and test.

**Methods.** FactCG-DeBERTa and MiniCheck (DeBERTa, RoBERTa) run through the library's own pipeline: the answer is split into claims, each claim is scored against every context window, a claim takes its best-supported window and the answer takes its weakest claim. Baselines are HHEM-2.1-Open, LettuceDetect (base, large), where the answer score is one minus the highest token hallucination probability, and an LLM judge served by Ollama using the library's judge prompt. On RAGTruth, LettuceDetect is given the dataset's original prompt, matching its training format. HHEM and LettuceDetect were trained on RAGTruth, so their RAGTruth results are in-distribution.

**Metrics.** AUROC with a 95% bootstrap interval (1,000 resamples) is the headline because it does not depend on a threshold. Balanced accuracy and F1 of the unsupported class use the threshold that maximises balanced accuracy on the dev split of the same dataset. LLM-AggreFact follows the benchmark protocol: balanced accuracy per constituent dataset at a fixed 0.5 threshold, and with a threshold tuned per dataset on `dev`, averaged over datasets. Failed or unparseable scores are kept and scored 0.5 rather than dropped, and their count is reported.

**Cascade.** The judge is run on the RAGTruth QA test split and a nested sample of 50 answers per RAGBench subset. The cascade replaces the verifier score with the judge score only when the verifier falls inside an uncertainty band, and is evaluated on answers scored by both, reporting the share escalated, AUROC and mean latency.

**Selective answering.** Risk–coverage curves show the unsupported rate among accepted answers as more answers are accepted in order of score, pooled over all test sets scored by every method shown.

## Retrieval quality

RAGBench marks which context sentences are relevant to each question. A document is labelled relevant if any of its sentences are.

- **Chunk level:** AUROC of each scorer for separating relevant from irrelevant documents of the same question.
- **Query level:** AUROC of the highest document score for separating questions that have relevant context from those that do not. Negatives are questions whose retrieved documents contain nothing relevant, plus each test question paired with the documents of a question from a distant subset.
- **Scorers:** the cross-encoder reranker with a plain sigmoid and with Platt scaling fitted on 300 training questions per subset, against raw cosine similarity from three embedding models, using each model's documented query and passage prefixes.
- **Calibration:** expected calibration error (15 bins) for the reranker probabilities, and the best dev-set threshold per scorer, which shows how far a fixed similarity cutoff moves between embedding models.

## Reproducing

All runs are resumable and skip examples already in the results files. `benchmark.ipynb` runs the full sequence on a Kaggle T4. Locally:

```bash
pip install -e ".[bench]"
python -m evaluation.run_retrieval
python -m evaluation.calibrate_retrieval
python -m evaluation.run_faithfulness --method factcg
python -m evaluation.run_faithfulness --method factcg --datasets aggrefact --limit 0 --dev-limit 300
python -m evaluation.report
```

LLM-AggreFact is gated on Hugging Face: accept its terms and set `HF_TOKEN` before running it. `make_parity_fixtures.py` regenerates the reference outputs used by `tests/test_parity.py`, and `measure_latency.py` times each component per query.
