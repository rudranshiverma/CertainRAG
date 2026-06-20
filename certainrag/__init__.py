#CertainRAG - Runtime uncertainty quantification for RAG pipelines.
import time
from typing import Optional
from certainrag.signals.retrieval_score import RetrievalConfidenceSignal
from certainrag.signals.faithfulness import FaithfulnessSignal
from certainrag.signals.semantic_entropy import SemanticEntropySignal
from certainrag.scorer import CompositeScorer, UncertaintyResult
from certainrag.exceptions import InputValidationError

__version__="0.1.0"
__all__=["CertainRAG", "UncertaintyResult"]


class CertainRAG:
    def __init__(
        self,
        faithfulness_model:str="cross-encoder/nli-deberta-v3-base",
        weights:tuple=(0.33, 0.33, 0.34),
        threshold:float=0.5,**kwargs):
        self._faithfulness=FaithfulnessSignal(model_name=faithfulness_model)
        self._entropy=SemanticEntropySignal(model_name=faithfulness_model)
        self._retrieval=RetrievalConfidenceSignal()
        self._scorer=CompositeScorer(weights=weights, threshold=threshold)

    def evaluate(self,query:str,answer:str,context_chunks:list[str],retrieval_scores:list[float],responses:Optional[list[str]]=None) -> UncertaintyResult:
        if not query or not query.strip():
            raise InputValidationError("query cannot be empty.")
        start=time.time()

        # Signal 1- retrieval confidence
        retrieval_result=self._retrieval.compute(retrieval_scores)

        # Signal 2- faithfulness
        faithfulness_result=self._faithfulness.compute(answer=answer,context_chunks=context_chunks,retrieval_scores=retrieval_scores)

        # Signal 3- semantic entropy (optional)
        if responses is not None:
            self._entropy._pipeline =self._faithfulness.get_pipeline()
            entropy_result=self._entropy.compute(responses)
            semantic_entropy=entropy_result["semantic_entropy"]
        else:
            semantic_entropy=0.5

        latency_ms=(time.time()-start)*1000

        return self._scorer.score(
            retrieval_confidence=retrieval_result["retrieval_confidence"],
            faithfulness_score=faithfulness_result["faithfulness_score"],
            semantic_entropy=semantic_entropy,
            supporting_chunks=faithfulness_result["supporting_chunks"],
            contradicting_chunks=faithfulness_result["contradicting_chunks"],
            latency_ms=latency_ms
        )

    def evaluate_entropy_only(self,responses: list[str])->dict:
        self._entropy._pipeline=self._faithfulness.get_pipeline()
        return self._entropy.compute(responses)