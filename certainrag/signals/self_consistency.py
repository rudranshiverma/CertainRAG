from .. import utils
from ..exceptions import BackendError, ConfigurationError

class SelfConsistencySignal:
    def __init__(self, llm_client, n_samples=3, temperature=1.0):
        if llm_client is None:
            raise ConfigurationError("SelfConsistencySignal requires an llm_client.")
        self.llm_client=llm_client
        self.n_samples=n_samples
        self.temperature=temperature
        self.embedder=utils.get_embedder()

    def score(self, question):
        try:
            samples=self.llm_client.generate(question, temperature=self.temperature, n=self.n_samples)
        except Exception as e:
            raise BackendError(f"Self-consistency resampling failed: {e}") from e

        samples=[s for s in samples if s and s.strip()]
        if len(samples)<2:
            return 0.0
        vectors=self.embedder.encode(samples, normalize_embeddings=True)

        distances=[]
        n=len(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                similarity=0.0
                for k in range(len(vectors[i])):
                    similarity+=vectors[i][k]*vectors[j][k]
                distance=(1.0-similarity)/2.0
                distances.append(distance)
        mean_distance=sum(distances)/len(distances)
        return max(0.0, min(1.0, mean_distance))