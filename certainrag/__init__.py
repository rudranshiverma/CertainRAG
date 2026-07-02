from .exceptions import ConfigurationError
from .llm_client import LLMClient, OllamaClient
from .scorer import CertainRAGResult, combine_scores, DEFAULT_WEIGHTS
from .signals import (
    RetrievalConfidenceSignal,
    FaithfulnessSignal,
    SelfConsistencySignal,
)
class CertainRAG:
    def __init__(
        self,
        llm_client=None,
        judge_llm_client=None,
        use_self_consistency=False,
        n_samples=3,
        self_consistency_temperature=1.0,
        weights=None,
    ):
        if llm_client is None:
            raise ConfigurationError(
                "CertainRAG requires an llm_client (used as the faithfulness judge "
                "unless judge_llm_client is set separately)."
            )
        self.llm_client=llm_client
        self.weights=weights or DEFAULT_WEIGHTS
        self.retrieval_signal=RetrievalConfidenceSignal()
        self.faithfulness_signal=FaithfulnessSignal(judge_llm_client or llm_client)
        self.use_self_consistency=use_self_consistency
        self.self_consistency_signal=None
        if use_self_consistency:
            self.self_consistency_signal = SelfConsistencySignal(
                llm_client, n_samples=n_samples, temperature=self_consistency_temperature
            )

    def score(self, question, answer, chunks, retrieval_scores=None):
        if not retrieval_scores:
            raise ConfigurationError(
                "retrieval_scores is required. Pass the similarity scores your "
                "retriever returned alongside the chunks. These must be cosine "
                "similarities in [0, 1], in the same order as chunks."
            )
        if not chunks:
            raise ConfigurationError(
                "chunks is required. Pass the retrieved context chunks your "
                "retriever returned for this query."
            )
        retrieval_confidence=self.retrieval_signal.score(retrieval_scores)
        faithfulness, faithfulness_reasoning=self.faithfulness_signal.score(question, answer, chunks)
        self_consistency=None
        if self.self_consistency_signal is not None:
            self_consistency=self.self_consistency_signal.score(question)

        uncertainty=combine_scores(
            retrieval_confidence=retrieval_confidence,
            faithfulness=faithfulness,
            self_consistency=self_consistency,
            weights=self.weights,
        )

        return CertainRAGResult(
            question=question,
            answer=answer,
            retrieval_confidence=retrieval_confidence,
            faithfulness=faithfulness,
            faithfulness_reasoning=faithfulness_reasoning,
            self_consistency=self_consistency,
            uncertainty=uncertainty,
        )
__all__ = ["CertainRAG", "CertainRAGResult","LLMClient", "OllamaClient"]