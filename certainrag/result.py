from dataclasses import asdict, dataclass, field
from enum import Enum


class Decision(str, Enum):
    ACCEPT = "accept"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNFAITHFUL = "unfaithful"


@dataclass
class RetrievalResult:
    score: float
    chunk_scores: list = field(default_factory=list)


@dataclass
class ClaimScore:
    text: str
    score: float


@dataclass
class FaithfulnessResult:
    score: float = None
    method: str = None
    verifier_score: float = None
    claims: list = field(default_factory=list)
    reasoning: str = None
    judge_error: str = None


@dataclass
class CertainRAGResult:
    decision: Decision
    retrieval: RetrievalResult
    faithfulness: FaithfulnessResult = None

    def to_dict(self):
        return asdict(self)
