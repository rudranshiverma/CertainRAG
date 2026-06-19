## faithfulness.py
- response sentences split using RE before NLI scoring because long answers may contain multiple claims. sentence-level scoring prevents dilution of per-claim entailment signal
- mean of sentence scores per chunk, then weighted average across chunks

## semantic_entropy.py
- used upper triangle only because diagnol is always 1 (every response compared with itself) and lower triangle contains the same pairs as upper triangle
- manual normalization done instead of 'faiss.normalize_L2' because we need full similarity matrix, not just top_k search results

##CertainRAG
- 