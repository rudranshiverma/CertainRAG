import pytest
from certainrag.signals.faithfulness import FaithfulnessSignal
from certainrag.exceptions import InputValidationError

signal=FaithfulnessSignal()

def test_faithful_answer():
    result=signal.compute(
        answer="Atrial fibrillation is caused by irregular electrical signals.",
        context_chunks=["Atrial fibrillation is caused by disorganized electrical signals."],
        retrieval_scores=[0.85]
    )
    assert result["faithfulness_score"]>0.5
    assert len(result["supporting_chunks"])>=1

def test_hallucinated_answer():
    result=signal.compute(
        answer="Atrial fibrillation is caused by brain signal misfiring.",
        context_chunks=["Photosynthesis converts sunlight into glucose in plant cells."],
        retrieval_scores=[0.78]
    )
    assert result["faithfulness_score"]<0.3

def test_empty_answer():
    with pytest.raises(InputValidationError):
        signal.compute(
            answer="",
            context_chunks=["some context"],
            retrieval_scores=[0.8])

def test_empty_chunks():
    with pytest.raises(InputValidationError):
        signal.compute(
            answer="some answer",
            context_chunks=[],
            retrieval_scores=[])

def test_mismatched_lengths():
    with pytest.raises(InputValidationError):
        signal.compute(
            answer="some answer",
            context_chunks=["chunk1", "chunk2"],
            retrieval_scores=[0.8])

def test_zero_weights():
    with pytest.raises(InputValidationError):
        signal.compute(
            answer="some answer",
            context_chunks=["chunk1"],
            retrieval_scores=[0.0])