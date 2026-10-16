import numpy as np

from ..exceptions import BackendError, ConfigurationError
from ..result import ClaimScore, FaithfulnessResult
from ..utils import pack_windows, split_sentences
from .judge import LLMJudge
from .verifiers import Verifier

MIN_CLAIM_WORDS = 4
SHORT_ANSWER_WORDS = 6
DEFAULT_ESCALATE_BAND = (0.35, 0.65)


def build_claims(question, answer):
    sentences = split_sentences(answer)
    if not sentences:
        return []
    if len(answer.split()) < SHORT_ANSWER_WORDS:
        return [f"{question.strip()} {answer.strip()}"]
    claims, buffer = [], ""
    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip()
        if len(buffer.split()) >= MIN_CLAIM_WORDS:
            claims.append(buffer)
            buffer = ""
    if buffer:
        if claims:
            claims[-1] = f"{claims[-1]} {buffer}"
        else:
            claims.append(buffer)
    return claims


class FaithfulnessSignal:
    def __init__(self, verifier="factcg", device=None, judge=None, escalate_band=DEFAULT_ESCALATE_BAND, batch_size=16):
        low, high = escalate_band
        if not 0.0 <= low <= high <= 1.0:
            raise ConfigurationError("escalate_band must satisfy 0 <= low <= high <= 1.")
        self.verifier = Verifier(verifier, device, batch_size) if isinstance(verifier, str) else verifier
        self.judge = LLMJudge(judge) if judge is not None else None
        self.escalate_band = escalate_band

    def score(self, question, answer, chunks):
        claims = build_claims(question, answer)
        if not claims:
            return FaithfulnessResult()
        claims = self.verifier.prepare_claims(claims)
        budget = self.verifier.window_budget(claims)
        windows = pack_windows(split_sentences("\n".join(chunks)), self.verifier.tokenizer, budget)
        if not windows:
            return FaithfulnessResult(
                score=0.0, method="verifier", verifier_score=0.0, claims=[ClaimScore(c, 0.0) for c in claims]
            )
        pairs = [(w, c) for c in claims for w in windows]
        probs = np.array(self.verifier.score([w for w, _ in pairs], [c for _, c in pairs]))
        claim_scores = probs.reshape(len(claims), len(windows)).max(axis=1)
        verifier_score = float(claim_scores.min())
        result = FaithfulnessResult(
            score=verifier_score,
            method="verifier",
            verifier_score=verifier_score,
            claims=[ClaimScore(c, float(s)) for c, s in zip(claims, claim_scores, strict=True)],
        )
        low, high = self.escalate_band
        if self.judge is None or not low <= verifier_score <= high:
            return result
        try:
            judge_score, reasoning = self.judge.score(question, answer, chunks)
        except BackendError as e:
            result.judge_error = str(e)
            return result
        if judge_score is None:
            result.judge_error = "Judge output had no valid SCORE line."
            result.reasoning = reasoning
            return result
        result.score, result.method, result.reasoning = judge_score, "judge", reasoning
        return result
