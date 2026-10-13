from dataclasses import dataclass, field

from .result import CertainRAGResult, Decision

DEFAULT_ABSTAIN_MESSAGE = "I don't have enough reliable information to answer that."


@dataclass
class Policy:
    retrieval_threshold: float = 0.5
    faithfulness_threshold: float = 0.5

    def retrieval_ok(self, retrieval):
        return retrieval.score >= self.retrieval_threshold

    def decide(self, retrieval, faithfulness=None):
        if not self.retrieval_ok(retrieval):
            return Decision.INSUFFICIENT_CONTEXT
        if faithfulness is None:
            return Decision.ACCEPT
        if faithfulness.score is None or faithfulness.score < self.faithfulness_threshold:
            return Decision.UNFAITHFUL
        return Decision.ACCEPT


@dataclass
class GuardOutput:
    answer: str
    result: CertainRAGResult
    abstained: bool
    steps: list = field(default_factory=list)


class Guard:
    def __init__(
        self,
        rag,
        retrieve,
        generate,
        fallback_retrieve=None,
        max_regenerations=1,
        abstain_message=DEFAULT_ABSTAIN_MESSAGE,
    ):
        self.rag = rag
        self.retrieve = retrieve
        self.generate = generate
        self.fallback_retrieve = fallback_retrieve
        self.max_regenerations = max_regenerations
        self.abstain_message = abstain_message

    def run(self, question):
        policy = self.rag.policy
        steps = ["retrieve"]
        chunks = self.retrieve(question)
        retrieval = self.rag.score_retrieval(question, chunks)
        if not policy.retrieval_ok(retrieval) and self.fallback_retrieve is not None:
            steps.append("fallback_retrieve")
            chunks = self.fallback_retrieve(question)
            retrieval = self.rag.score_retrieval(question, chunks)
        if not policy.retrieval_ok(retrieval):
            result = CertainRAGResult(Decision.INSUFFICIENT_CONTEXT, retrieval)
            return GuardOutput(self.abstain_message, result, True, steps)
        for attempt in range(self.max_regenerations + 1):
            strict = attempt > 0
            steps.append("regenerate_strict" if strict else "generate")
            answer = self.generate(question, chunks, strict)
            faithfulness = self.rag.score_faithfulness(question, answer, chunks)
            result = CertainRAGResult(policy.decide(retrieval, faithfulness), retrieval, faithfulness)
            if result.decision == Decision.ACCEPT:
                return GuardOutput(answer, result, False, steps)
        return GuardOutput(self.abstain_message, result, True, steps)
