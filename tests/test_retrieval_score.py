import pytest
from certainrag.signals.retrieval_score import RetrievalConfidenceSignal
from certainrag.exceptions import InputValidationError

signal=RetrievalConfidenceSignal()

def test_computation():
    result=signal.compute([0.8, 0.6, 0.7])
    assert "retrieval_confidence" in result
    assert 0.0<=result["retrieval_confidence"]<=1.0
    assert abs(result["retrieval_confidence"]-0.7)<0.001

def test_single_score():
    result=signal.compute([0.9])
    assert result["retrieval_confidence"]==0.9

def test_empty_raises():
    with pytest.raises(InputValidationError):
        signal.compute([])

def test_invalid_score():
    with pytest.raises(InputValidationError):
        signal.compute([0.5, 1.5])

def test_negative_score():
    with pytest.raises(InputValidationError):
        signal.compute([-0.1, 0.5])

def test_boundary_values():
    result=signal.compute([0.0, 1.0])
    assert result["retrieval_confidence"]==0.5