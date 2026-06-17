from certainrag.faithfulness import FaithfulnessScorer

scorer=FaithfulnessScorer()

#test 1- faithful answer
chunks_faithful=[
    {
        "chunk":"Atrial fibrillation is caused by disorganized electrical signals in the heart's upper chambers.",
        "similarity_score":0.85
    },
    {
        "chunk":"Risk factors for atrial fibrillation include high blood pressure and heart disease.",
        "similarity_score":0.72
    }
]
answer_faithful = "Atrial fibrillation is caused by irregular electrical signals in the upper chambers of the heart."
result_faithful = scorer.score(answer_faithful, chunks_faithful)
print("=== Test 1: Faithful Answer ===")
print(f"Faithfulness Score: {result_faithful['faithfulness_score']:.4f}")
print(f"Supporting chunks: {len(result_faithful['supporting_chunks'])}")
print(f"Contradicting chunks: {len(result_faithful['contradicting_chunks'])}")
print(f"Sentences evaluated: {result_faithful['sentences_evaluated']}")
print()

#test 2- hallucinated answer
chunks_hallucinated=[
    {
        "chunk":"Photosynthesis converts sunlight into glucose using chlorophyll in plant cells.",
        "similarity_score":0.78
    }
]
answer_hallucinated="Atrial fibrillation is a neurological disorder caused by brain signal misfiring."
result_hallucinated=scorer.score(answer_hallucinated, chunks_hallucinated)
print("=== Test 2: Hallucinated Answer ===")
print(f"Faithfulness Score: {result_hallucinated['faithfulness_score']:.4f}")
print(f"Supporting chunks: {len(result_hallucinated['supporting_chunks'])}")
print(f"Contradicting chunks: {len(result_hallucinated['contradicting_chunks'])}")
print()

#test 3- multisentence answer
chunks_multi=[
    {
        "chunk": "The mitochondria produces ATP through cellular respiration. It is found in eukaryotic cells.",
        "similarity_score":0.90
    }
]
answer_multi="Mitochondria produces energy in the form of ATP. It is present in plant and animal cells. It was discovered on Mars in 1969."

result_multi=scorer.score(answer_multi, chunks_multi)
print("=== Test 3: Multi-sentence with one hallucinated claim ===")
print(f"Faithfulness Score: {result_multi['faithfulness_score']:.4f}")
print(f"Sentences evaluated: {result_multi['sentences_evaluated']}")
print(f"Per-sentence scores: {result_multi['chunk_scores'][0]['sentence_scores']}")