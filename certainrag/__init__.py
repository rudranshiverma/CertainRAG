from .exceptions import BackendError, CertainRAGError, ConfigurationError
from .policy import Guard, GuardOutput, Policy
from .result import CertainRAGResult, ClaimScore, Decision, FaithfulnessResult, RetrievalResult
from .signals import VERIFIERS, FaithfulnessSignal, RetrievalSignal
from .signals.faithfulness import DEFAULT_ESCALATE_BAND
from .signals.retrieval_score import DEFAULT_PLATT, DEFAULT_RERANKER

__version__ = "0.2.0"


class CertainRAG:
    def __init__(
        self,
        verifier="factcg",
        reranker=DEFAULT_RERANKER,
        device=None,
        judge=None,
        escalate_band=DEFAULT_ESCALATE_BAND,
        policy=None,
        retrieval_platt=DEFAULT_PLATT,
    ):
        self.policy = policy or Policy()
        self.retrieval = RetrievalSignal(reranker, device, retrieval_platt) if isinstance(reranker, str) else reranker
        self.faithfulness = FaithfulnessSignal(verifier, device, judge, escalate_band)

    def score_retrieval(self, question, chunks):
        return self.retrieval.score(question, chunks)

    def score_faithfulness(self, question, answer, chunks):
        return self.faithfulness.score(question, answer, chunks)

    def score(self, question, answer, chunks):
        retrieval = self.score_retrieval(question, chunks)
        if not self.policy.retrieval_ok(retrieval):
            return CertainRAGResult(Decision.INSUFFICIENT_CONTEXT, retrieval)
        faithfulness = self.score_faithfulness(question, answer, chunks)
        return CertainRAGResult(self.policy.decide(retrieval, faithfulness), retrieval, faithfulness)


__all__ = [
    "VERIFIERS",
    "BackendError",
    "CertainRAG",
    "CertainRAGError",
    "CertainRAGResult",
    "ClaimScore",
    "ConfigurationError",
    "Decision",
    "FaithfulnessResult",
    "Guard",
    "GuardOutput",
    "Policy",
    "RetrievalResult",
]