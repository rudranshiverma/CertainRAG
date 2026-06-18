import time
from certainrag.retriever import Retriever
from certainrag.faithfulness import FaithfulnessScorer
from certainrag.semantic_entropy import SemanticEntropyScorer
from certainrag.scorer import CompositeScorer, UncertaintyResult

class CertainRag:
    def __init__(self, method="composite", threshold=0.5, fast_mode=False,weights=(0.33,0.33,0.34)):
        self.method=method
        self.faithfulness_scorer=FaithfulnessScorer()
        self.entropy_scorer=SemanticEntropyScorer(fast_mode=fast_mode)
        self.composite_scorer=CompositeScorer(weights=weights,threshold=threshold)
    def evaluate(self,query:str,answer:str,context_chunks:list[str],retrieval_scores:list[float]=None)->UncertaintyResult:
        start=time.time()

        #first signal = retrieval confidence
        if retrieval_scores:
            retrieval_confidence=sum(retrieval_scores)/len(retrieval_scores)
        else:
            retrieval_confidence=0.5

        #second signal = faithfulness
        chunks_with_scores=[
            {"chunk":c, "similarity_score":s}
            for c,s in zip(context_chunks, retrieval_scores or [0.5]*len(context_chunks))
        ]
        faith_result=self.faithfulness_scorer.score(answer, chunks_with_scores)

        #third signal = semantic entropy
        entropy_result=self.entropy_scorer.score(query,context_chunks)

        latency_ms=(time.time()-start)*1000

        return self.composite_scorer.score(
            retrieval_confidence=retrieval_confidence,
            faithfulness_score=faith_result["faithfulness_score"],
            semantic_entropy=entropy_result["semantic_entropy"],
            supporting_chunks=faith_result["supporting_chunks"],
            contradicting_chunks=faith_result["contradicting_chunks"],
            latency_ms=latency_ms
        )
