import numpy as np

class SimpleRetriever:
    def __init__(self, embedder, chunks):
        self.embedder = embedder
        self.chunks = chunks
        self.index = None
        self._build_index()

    def _build_index(self):
        import faiss
        vectors = self.embedder.encode(self.chunks, normalize_embeddings=True, convert_to_numpy=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors.astype(np.float32))

    def retrieve(self, query, top_k=4):
        k=min(top_k, len(self.chunks))
        if k==0:
            return [], []

        query_vector = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores,indices = self.index.search(query_vector.astype(np.float32), k)
        scores,indices = scores[0], indices[0]

        result_chunks =[]
        result_scores=[]
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.chunks):
                result_chunks.append(self.chunks[idx])
                result_scores.append(float(score))

        return result_chunks, result_scores