from sentence_transformers import SentenceTransformers
import faiss
import numpy

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model=SentenceTransformers(model_name)
        self.index=None
        self.chunks=[]
    def index_documents(self,chunks:list[str]):
        self.chunks=chunks
        embeddings=self.model.encode(chunks, convert_to_numpy=True)
        dimension=embeddings.shape[1]
        self.index=faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    def retrieve(self, query:str, top_k:int=4):
        query_embedding=self.model.encode([query],convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores,indices=self.index.search(query_embedding,top_k)
        results=[]
        for score,idx in zip(scores[0],indices[0]):
            results.append({
                "chunk":self.chunks[idx],
                "similarity_score":float(score)
            })
        return results