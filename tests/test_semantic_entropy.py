import pytest
from certainrag.signals.semantic_entropy import SemanticEntropySignal
from certainrag.exceptions import InputValidationError

signal = SemanticEntropySignal()

def test_low_entropy():
    #All responses paraphrase the same meaning - should collapse to ~1 cluster
    responses = [
        "Photosynthesis converts sunlight into glucose in plants.",
        "Plants use sunlight to produce glucose through photosynthesis.",
        "Through photosynthesis, plants make glucose using light energy.",
        "Sunlight is converted to glucose by plants via photosynthesis.",
        "Plants synthesize glucose from sunlight through photosynthesis."
    ]
    result=signal.compute(responses)
    print(f"\nLOW ENTROPY TEST — entropy: {result['semantic_entropy']:.4f}, clusters: {result['n_clusters']}, sizes: {result['cluster_sizes']}")
    assert result["n_clusters"]<=2
    assert result["semantic_entropy"]<0.4

def test_high_entropy():
    #Responses with genuinely different, non-entailing claims.
    responses = [
        "Photosynthesis converts sunlight into glucose in plant cells.",
        "I don't have enough information to answer this question.",
        "The mitochondria is the powerhouse of the cell.",
        "This question is about quantum mechanics and entanglement.",
        "Plants need water and minerals from soil to grow tall."
    ]
    result=signal.compute(responses)
    print(f"\nHIGH ENTROPY TEST — entropy: {result['semantic_entropy']:.4f}, clusters: {result['n_clusters']}, sizes: {result['cluster_sizes']}")
    assert result["n_clusters"] >= 3
    assert result["n_clusters"] >= 3
    assert result["semantic_entropy"] > 0.5

def test_single_response_raises():
    with pytest.raises(InputValidationError):
        signal.compute(["only one response"])

def test_empty_list_raises():
    with pytest.raises(InputValidationError):
        signal.compute([])

def test_empty_strings_raises():
    with pytest.raises(InputValidationError):
        signal.compute(["", "   ", ""])

def test_cluster_sizes_sum_to_n():
    responses = [
        "The sky is blue during the day.",
        "Daytime skies appear blue.",
        "Grass is green in most climates."
    ]
    result=signal.compute(responses)
    assert sum(result["cluster_sizes"]) == len(responses)

def test_output_has_required_fields():
    responses= [
        "Water boils at 100 degrees Celsius at sea level.",
        "At sea level, water reaches boiling point at 100°C."
    ]
    result=signal.compute(responses)
    assert "semantic_entropy" in result
    assert "n_clusters" in result
    assert "cluster_sizes" in result
    assert "clusters" in result
    assert 0.0 <= result["semantic_entropy"]<=1.0