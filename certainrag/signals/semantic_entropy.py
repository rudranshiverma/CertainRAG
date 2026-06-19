from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_ollama import ChatOllama

class SemanticEntropyScorer:
    def __init__(self, n_samples=5, temperature=1.0, model_name="all-MiniLM-L6-v2", fast_mode=False):
        self.n_samples=3 if fast_mode else n_samples
        self.temperature=temperature
        self.embedder=SentenceTransformer(model_name)
        self.llm=ChatOllama(model="mistral", temperature=temperature)
    def score(self, query:str,context_chunks:list[str])->dict:
        context="\n\n".join(context_chunks)
        prompt=f"""Answer the question using only the context provided.
        Context:{context}
        Question:{query}
        Answer:"""
        responses=[]
        for _ in range(self.n_samples):
            response=self.llm.invoke(prompt)
            responses.append(response.content)
        embeddings=self.embedder.encode(responses,convert_to_numpy=True)

        #pairwise cosine similarity matrix
        norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
        normalized=embeddings/norms
        similarity_matrix=np.dot(normalized,normalized.T)

        n=len(responses)
        upper_triangle=[
            similarity_matrix[i][j]
            for i in range(n)
            for j in range(i+1,n)
        ]
        mean_similarity=float(np.mean(upper_triangle))
        semantic_entropy=1.0-mean_similarity

        return{
            "semantic_entropy":semantic_entropy,
            "mean_similarity": mean_similarity,
            "responses":responses,
            "n_samples":self.n_samples
        }