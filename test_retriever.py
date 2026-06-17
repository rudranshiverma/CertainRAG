from certainrag.retriever import Retriever

chunks=[
    "The mitochondria is the powerhouse of the cell.",
    "Photosynthesis converts sunlight into glucose in plants.",
    "Neural networks are inspired by the human brain.",
    "RAG systems retrieve documents before generating answers.",
    "The heart pumps blood through the circulatory system."
]

retriever=Retriever()
retriever.index_documents(chunks)
query="How do AI systems work?"
results=retriever.retrieve(query,top_k=2)

print(f"Query: {query}\n")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f" Chunk: {result['chunk']}")
    print(f" Similarity Score: {result['similarity_score']:.4f}")