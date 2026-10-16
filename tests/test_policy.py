import json

import pytest

from certainrag import CertainRAG, Decision, FaithfulnessResult, Guard, Policy, RetrievalResult

ANSWER = "Aspirin reduced mortality by 12 % in the trial."
CHUNKS = ["Aspirin reduced mortality by 12 % in the trial."]


@pytest.mark.parametrize(
    "retrieval, faithfulness, expected",
    [
        (0.2, 0.9, Decision.INSUFFICIENT_CONTEXT),
        (0.8, None, Decision.ACCEPT),
        (0.8, 0.9, Decision.ACCEPT),
        (0.8, 0.5, Decision.ACCEPT),
        (0.8, 0.4, Decision.UNFAITHFUL),
    ],
)
def test_decide(retrieval, faithfulness, expected):
    f = None if faithfulness is None else FaithfulnessResult(score=faithfulness)
    assert Policy().decide(RetrievalResult(retrieval), f) == expected


def test_empty_answer_is_not_accepted():
    assert Policy().decide(RetrievalResult(0.9), FaithfulnessResult()) == Decision.UNFAITHFUL


def rag(fake_retrieval, verifier, retrieval=0.8, **kwargs):
    return CertainRAG(verifier=verifier, reranker=fake_retrieval(retrieval), **kwargs)


def test_score_accepts(fake_retrieval, fake_verifier):
    result = rag(fake_retrieval, fake_verifier).score("q?", ANSWER, CHUNKS)
    assert result.decision == Decision.ACCEPT
    assert json.loads(json.dumps(result.to_dict()))["decision"] == "accept"


def test_score_skips_faithfulness_when_retrieval_fails(fake_retrieval, fake_verifier):
    result = rag(fake_retrieval, fake_verifier, retrieval=0.1).score("q?", ANSWER, CHUNKS)
    assert result.decision == Decision.INSUFFICIENT_CONTEXT
    assert result.faithfulness is None
    assert fake_verifier.calls == 0


def test_score_flags_unfaithful(fake_retrieval, constant_verifier):
    result = rag(fake_retrieval, constant_verifier(0.1)).score("q?", ANSWER, CHUNKS)
    assert result.decision == Decision.UNFAITHFUL


class Generator:
    def __init__(self):
        self.strict_flags = []

    def __call__(self, question, chunks, strict):
        self.strict_flags.append(strict)
        return ANSWER


def test_guard_regenerates_strict_then_accepts(fake_retrieval, constant_verifier):
    gen = Generator()
    out = Guard(rag(fake_retrieval, constant_verifier(0.1, 0.9)), lambda q: CHUNKS, gen).run("q?")
    assert not out.abstained
    assert out.answer == ANSWER
    assert out.steps == ["retrieve", "generate", "regenerate_strict"]
    assert gen.strict_flags == [False, True]


def test_guard_abstains_after_regenerations(fake_retrieval, constant_verifier):
    gen = Generator()
    out = Guard(rag(fake_retrieval, constant_verifier(0.1)), lambda q: CHUNKS, gen, max_regenerations=2).run("q?")
    assert out.abstained
    assert out.result.decision == Decision.UNFAITHFUL
    assert gen.strict_flags == [False, True, True]


def test_guard_uses_fallback_retrieval(constant_verifier):
    class Retrieval:
        def score(self, question, chunks):
            return RetrievalResult(0.9 if chunks == ["web result"] else 0.1)

    gen = Generator()
    guard = Guard(
        CertainRAG(verifier=constant_verifier(0.9), reranker=Retrieval()),
        lambda q: CHUNKS,
        gen,
        fallback_retrieve=lambda q: ["web result"],
    )
    out = guard.run("q?")
    assert not out.abstained
    assert out.steps == ["retrieve", "fallback_retrieve", "generate"]
    assert gen.strict_flags == [False]


def test_guard_abstains_without_generating_on_bad_retrieval(fake_retrieval, fake_verifier):
    gen = Generator()
    out = Guard(rag(fake_retrieval, fake_verifier, retrieval=0.1), lambda q: CHUNKS, gen).run("q?")
    assert out.abstained
    assert out.result.decision == Decision.INSUFFICIENT_CONTEXT
    assert gen.strict_flags == []
