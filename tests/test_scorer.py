from certainrag.scorer import CompositeScorer

scorer = CompositeScorer()

#test 1-all signals confident, should be LOW uncertainty
result_low=scorer.score(
    retrieval_confidence=0.85,
    faithfulness_score=0.90,
    semantic_entropy=0.05,
    supporting_chunks=["chunk1", "chunk2"],
    contradicting_chunks=[],
    latency_ms=1200.0
)
print("=== Test 1: Low Uncertainty ===")
print(f"Uncertainty Score: {result_low.uncertainty_score}")
print(f"Uncertainty Level: {result_low.uncertainty_level}")
print(f"Should Abstain: {result_low.should_abstain}")
print(f"Explanation: {result_low.explanation}")
print()

#test 2 — all signals uncertain, should be HIGH uncertainty
result_high=scorer.score(
    retrieval_confidence=0.21,
    faithfulness_score=0.15,
    semantic_entropy=0.42,
    supporting_chunks=[],
    contradicting_chunks=["chunk1"],
    latency_ms=1500.0
)
print("=== Test 2: High Uncertainty ===")
print(f"Uncertainty Score: {result_high.uncertainty_score}")
print(f"Uncertainty Level: {result_high.uncertainty_level}")
print(f"Should Abstain: {result_high.should_abstain}")
print(f"Explanation: {result_high.explanation}")
print()

#test 2- mixed signals, should be MEDIUM
result_medium = scorer.score(
    retrieval_confidence=0.75,
    faithfulness_score=0.35,
    semantic_entropy=0.18,
    supporting_chunks=["chunk1"],
    contradicting_chunks=["chunk2"],
    latency_ms=1300.0
)
print("=== Test 3: Mixed Signals ===")
print(f"Uncertainty Score: {result_medium.uncertainty_score}")
print(f"Uncertainty Level: {result_medium.uncertainty_level}")
print(f"Should Abstain: {result_medium.should_abstain}")
print(f"Explanation: {result_medium.explanation}")