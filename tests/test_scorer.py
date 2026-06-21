import pytest
from certainrag.scorer import CompositeScorer, UncertaintyResult
from certainrag.exceptions import InputValidationError

def test_low_uncertainty():
    scorer=CompositeScorer()
    result=scorer.score(
        retrieval_confidence=0.90,
        faithfulness_score=0.88,
        semantic_entropy=0.04,
        supporting_chunks=["chunk1"],
        contradicting_chunks=[],
        latency_ms=1200.0
    )
    assert result.uncertainty_level=="LOW"
    assert result.should_abstain==False
    assert isinstance(result,UncertaintyResult)

def test_high_uncertainty():
    scorer=CompositeScorer()
    result=scorer.score(
        retrieval_confidence=0.20,
        faithfulness_score=0.15,
        semantic_entropy=0.70,
        supporting_chunks=[],
        contradicting_chunks=["chunk1"],
        latency_ms=1500.0
    )
    assert result.uncertainty_level=="HIGH"
    assert result.should_abstain==True

def test_medium_uncertainty():
    scorer=CompositeScorer()
    result=scorer.score(
        retrieval_confidence=0.65,
        faithfulness_score=0.45,
        semantic_entropy=0.20,
        supporting_chunks=["chunk1"],
        contradicting_chunks=["chunk2"],
        latency_ms=1300.0
    )
    assert result.uncertainty_level=="MEDIUM"

def test_invalid_weights():
    with pytest.raises(InputValidationError):
        CompositeScorer(weights=(0.5, 0.5, 0.5))

def test_invalid_threshold():
    with pytest.raises(InputValidationError):
        CompositeScorer(threshold=1.5)

def test_score_out_of_range():
    scorer=CompositeScorer()
    with pytest.raises(InputValidationError):
        scorer.score(
            retrieval_confidence=1.5,
            faithfulness_score=0.8,
            semantic_entropy=0.1,
            supporting_chunks=[],
            contradicting_chunks=[],
            latency_ms=100.0
        )

def test_result_fields_present():
    scorer=CompositeScorer()
    result=scorer.score(
        retrieval_confidence=0.7,
        faithfulness_score=0.7,
        semantic_entropy=0.1,
        supporting_chunks=[],
        contradicting_chunks=[],
        latency_ms=500.0
    )
    assert hasattr(result,"uncertainty_score")
    assert hasattr(result,"uncertainty_level")
    assert hasattr(result,"should_abstain")
    assert hasattr(result,"signal_breakdown")
    assert hasattr(result,"explanation")