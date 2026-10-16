import pytest
import torch

from certainrag.exceptions import BackendError, ConfigurationError
from certainrag.signals import verifiers
from certainrag.signals.faithfulness import FaithfulnessSignal, build_claims
from certainrag.signals.judge import LLMJudge
from certainrag.signals.verifiers import Verifier

CHUNKS = [
    "Aspirin reduced mortality by 12 % in the trial. The trial ran for two years in Boston.",
    "The weather in spring is not in this study.",
]
SUPPORTED = "Aspirin reduced mortality by 12 % in the trial. The trial ran in Boston."
ONE_WRONG = "Aspirin reduced mortality by 12 % in the trial. The trial ran in Paris."
# copied verbatim from the official FactCG repository (factcg/utils.py)
FACTCG_REFERENCE = (
    '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "{text_b}"?\n\n'
    "OPTIONS:\n- Yes\n- No\nI think the answer is "
)


def test_build_claims_short_answer_includes_question():
    assert build_claims("Is aspirin effective?", "Yes.") == ["Is aspirin effective? Yes."]


def test_build_claims_merges_fragments():
    claims = build_claims("q?", "Yes. Aspirin reduced mortality in the trial. It is cheap.")
    assert claims == ["Yes. Aspirin reduced mortality in the trial. It is cheap."]


def test_build_claims_keeps_lines_separate():
    assert build_claims("q?", "- reduces pain in adults\n- costs little money overall") == [
        "- reduces pain in adults",
        "- costs little money overall",
    ]


def test_build_claims_empty():
    assert build_claims("q?", "   ") == []


def test_supported_answer(fake_verifier):
    result = FaithfulnessSignal(fake_verifier).score("q?", SUPPORTED, CHUNKS)
    assert result.score == 0.9
    assert result.method == "verifier"
    assert [c.score for c in result.claims] == [0.9, 0.9]


def test_one_unsupported_claim_sets_score(fake_verifier):
    result = FaithfulnessSignal(fake_verifier).score("q?", ONE_WRONG, CHUNKS)
    assert result.score == 0.1
    assert [c.score for c in result.claims] == [0.9, 0.1]


def test_scores_all_pairs_in_one_call(fake_verifier):
    FaithfulnessSignal(fake_verifier).score("q?", SUPPORTED, CHUNKS)
    assert fake_verifier.calls == 1


def test_empty_answer_and_no_context(fake_verifier):
    signal = FaithfulnessSignal(fake_verifier)
    assert signal.score("q?", "", CHUNKS).score is None
    assert signal.score("q?", SUPPORTED, []).score == 0.0


def test_judge_used_inside_band(constant_verifier):
    prompts = []
    judge = lambda p: prompts.append(p) or "checked\nSCORE: 0.8"  # noqa: E731
    result = FaithfulnessSignal(constant_verifier(0.5), judge=judge).score("q?", SUPPORTED, CHUNKS)
    assert (result.score, result.method, result.verifier_score, result.reasoning) == (0.8, "judge", 0.5, "checked")
    assert len(prompts) == 1


def test_judge_skipped_outside_band(constant_verifier):
    calls = []
    result = FaithfulnessSignal(constant_verifier(0.9), judge=calls.append).score("q?", SUPPORTED, CHUNKS)
    assert result.method == "verifier"
    assert calls == []


def test_judge_failure_falls_back(constant_verifier):
    def broken(prompt):
        raise TimeoutError("timeout")

    result = FaithfulnessSignal(constant_verifier(0.5), judge=broken).score("q?", SUPPORTED, CHUNKS)
    assert (result.score, result.method) == (0.5, "verifier")
    assert "timeout" in result.judge_error


def test_unparseable_judge_falls_back(constant_verifier):
    result = FaithfulnessSignal(constant_verifier(0.5), judge=lambda p: "no score").score("q?", SUPPORTED, CHUNKS)
    assert (result.score, result.method) == (0.5, "verifier")
    assert result.judge_error


def test_invalid_band(constant_verifier):
    with pytest.raises(ConfigurationError):
        FaithfulnessSignal(constant_verifier(0.5), escalate_band=(0.7, 0.3))


@pytest.mark.parametrize(
    "output, expected",
    [
        ("reasoning\nSCORE: 0.85", 0.85),
        ("mentions SCORE: 0.2 first\nSCORE: 0.9", 0.9),
        ("score: .5", 0.5),
        ("SCORE: 1", 1.0),
        ("no score here", None),
        ("SCORE: 85", None),
    ],
)
def test_judge_parsing(output, expected):
    score, _ = LLMJudge(lambda p: output).score("q?", "a.", ["c1", "c2"])
    assert score == expected


def test_judge_prompt_numbers_chunks():
    prompts = []
    LLMJudge(lambda p: prompts.append(p) or "SCORE: 1").score("q?", "a.", ["c1", "c2"])
    assert "[1] c1\n[2] c2" in prompts[0]


def test_judge_errors():
    with pytest.raises(ConfigurationError):
        LLMJudge("not callable")
    with pytest.raises(BackendError):
        LLMJudge(lambda p: 1 / 0).score("q?", "a.", ["c"])


@pytest.fixture
def tiny_factcg(tiny_verifier, monkeypatch):
    monkeypatch.setitem(verifiers.VERIFIERS, "tiny-factcg", (tiny_verifier, "factcg", 512))
    monkeypatch.setitem(verifiers.VERIFIERS, "tiny-minicheck", (tiny_verifier, "minicheck", 512))


@pytest.mark.parametrize("name", ["tiny-factcg", "tiny-minicheck"])
def test_verifier_batching_is_consistent(tiny_factcg, name):
    v = Verifier(name, device="cpu", batch_size=2)
    windows, claims = CHUNKS * 3, [SUPPORTED, ONE_WRONG, "Yes."] * 2
    batched = v.score(windows, claims)
    single = [v.score([w], [c])[0] for w, c in zip(windows, claims, strict=True)]
    assert batched == pytest.approx(single, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in batched)


def test_verifier_formats(tiny_factcg):
    factcg = Verifier("tiny-factcg", device="cpu")
    assert factcg._format("doc", "claim") == FACTCG_REFERENCE.format(text_a="doc", text_b="claim")
    assert Verifier("tiny-minicheck", device="cpu")._format("doc", "claim") == "doc[SEP]claim"


@pytest.mark.parametrize("name", ["tiny-factcg", "tiny-minicheck"])
def test_verifier_returns_supported_class(tiny_factcg, name):
    v = Verifier(name, device="cpu")
    with torch.no_grad():
        v.model.classifier.weight.zero_()
        v.model.classifier.bias.copy_(torch.tensor([-5.0, 5.0]))
    assert v.score(CHUNKS, [SUPPORTED, ONE_WRONG]) == pytest.approx([1.0, 1.0], abs=1e-3)


def test_verifier_inputs_fit(tiny_factcg):
    v = Verifier("tiny-minicheck", device="cpu")
    claims = v.prepare_claims(["the " * 600, "aspirin reduced mortality ."])
    budget = v.window_budget(claims)
    assert budget >= verifiers.MIN_WINDOW_TOKENS
    window = " ".join(["the"] * budget)
    assert len(v.tokenizer(v._format(window, claims[0]))["input_ids"]) <= v.max_length


def test_unknown_verifier():
    with pytest.raises(ConfigurationError):
        Verifier("nope")
