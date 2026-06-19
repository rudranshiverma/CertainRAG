from certainrag.semantic_entropy import SemanticEntropyScorer

scorer=SemanticEntropyScorer()

#test 1- high entropy
chunks_high=[
        "Atrial fibrillation is caused by disorganized electrical signals in the heart's upper chambers.",
        "Risk factors for atrial fibrillation include high blood pressure and heart disease.",
]
query_high_entropy = "Can arm pain be a sign of heart attack?"
result_high_entropy = scorer.score(query_high_entropy, chunks_high)
print("=== Test 1: High Entropy ===")
print(f"Semantic Entropy: {result_high_entropy['semantic_entropy']:.4f}")
print(f"Mean Similarity: {result_high_entropy['mean_similarity']:.4f}")
print()

#test 2- low entropy
chunks_low=[
        "Photosynthesis allows green plants to convert sunlight into chemical energy stored in glucose. Chlorophyll inside the leaves absorbs sunlight and plays a major role in photosynthesis. During photosynthesis, plants take in carbon dioxide and release oxygen into the atmosphere.",
        "Plant diseases can spread rapidly through infected soil, water, or air.",
        "Yellowing leaves and stunted growth are common symptoms of diseased plants",
        "Wilting leaves even when the soil has enough water can indicate disease. Brown or black spots on leaves are common symptoms of fungal infections."
]
query_low_entropy="What are the symptoms of plant diseases?"
result_low_entropy=scorer.score(query_low_entropy, chunks_low)
print("=== Test 2: Low Entropy Answer ===")
print(f"Semantic Entropy: {result_low_entropy['semantic_entropy']:.4f}")
print(f"Mean Similarity: {result_low_entropy['mean_similarity']}")
print()