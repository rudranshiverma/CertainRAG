import numpy as np
from certainrag.exceptions import InputValidationError, ComputationError, ModelLoadError

#measures uncertainty by checking how consistent multiple LLM responses are in meaning
"""works in two modes:
1. fast_mode=False(default) - formal semantic entropy. clusters responses by birectional NLI entailment,computes Shannon entropy over cluster sizes.
it's accurate but slower - O(n^2) NLI calls for N responses.
2. fast_mode=True - computes pairwise cosine similarity between sentence embeddings of responses.
fast but it's an approximation, not true entropy.
This signal requires pre-generated responses which the developer must generate using their own LLM and pass them here.
"""
class SemanticEntropySignal:
    def __init__(self, nli_pipeline=None, model_name:str="cross-encoder/nli-deberta-v3-base", embedding_model_name:str="sentence-transformers/all-MiniLM-L6-v2", fast_mode:bool=False):
        self.model_name=model_name
        self.embedding_model_name=embedding_model_name
        self.fast_mode=fast_mode
        self._nli_pipeline=nli_pipeline
        self._embedder=None
    def _load_nli_model(self):
        if self._nli_pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._nli_pipeline=pipeline("text-classification",model=self.model_name,top_k=None)
        except Exception as e:
            raise ModelLoadError(f"Failed to load NLI model '{self.model_name}: {e}")
    def _load_embedding_model(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder=SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            raise ModelLoadError(f"Failed to load embedding model '{self.embedding_model_name}: {e}")
        
    def _entails(self,premise:str,hypothesis:str,threshold:float=0.5)->bool:
        try:
            result=self._nli_pipeline(f"{premise} [SEP] {hypothesis}")[0]
            for item in result:
                if item["label"].lower()=="entailment":
                    return item["score"] > threshold
            return False
        except Exception as e:
            raise ComputationError(f"NLI inference failed during clustering: {e}")
    def _cluster_by_entailment(self,responses:list[str])->list[list[int]]:
        n=len(responses)
        assigned=[False]*n
        clusters=[]
        for i in range(n):
            if assigned[i]:
                continue
            cluster=[i]
            assigned[i]=True
            for j in range(i+1,n):
                if assigned[j]:
                    continue
                forward=self._entails(responses[i],responses[j])
                backward=self._entails(responses[j],responses[i])
                if forward and backward:
                    cluster.append(j)
                    assigned[j]=True
            clusters.append(cluster)
        return clusters
    def _compute_entropy(self,responses:list[str])->dict:
        self._load_nli_model()
        clusters=self._cluster_by_entailment(responses)
        n=len(responses)
        cluster_sizes=[len(c) for c in clusters]
        probs=[size/n for size in cluster_sizes]
        entropy=-sum(p*np.log(p) for p in probs if p>0)
        max_entropy=np.log(n) if n>1 else 1.0
        normalized_entropy=float(entropy/max_entropy) if max_entropy>0 else 0.0
        return {
            "semantic_entropy":normalized_entropy,
            "n_clusters":len(clusters),
            "cluster_sizes":cluster_sizes,
            "clusters":[[responses[idx] for idx in c] for c in clusters],
            "n_responses":n,
            "mode":"true_entropy"
        }
    #fast mode:
    def _compute_embedding_dispersion(self,responses:list[str])->dict:
        self._load_embedding_model()
        embeddings=self._embedder.encode(responses,convert_to_numpy=True)
        norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
        norms=np.where(norms==0,1e-8,norms)
        normalized=embeddings/norms
        similarity_matrix=np.dot(normalized,normalized.T)
        n=len(responses)
        upper_triangle=[
            similarity_matrix[i][j]
            for i in range(n)
            for j in range(i+1,n)
        ]
        mean_similarity=float(np.mean(upper_triangle))
        mean_similarity=np.clip(mean_similarity,0.0,1.0)
        dispersion=np.clip(1.0-mean_similarity,0.0,1.0)
        return{
            "semantic_entropy":dispersion,
            "mean_similarity":mean_similarity,
            "n_responses":n,
            "mode":"fast_dispersion_proxy"
        }
    def compute(self, responses: list[str]) -> dict:
        if not responses:
            raise InputValidationError(
                "responses cannot be empty. Pass a list of pre-generated LLM responses to compute semantic entropy."
            )

        if len(responses) < 2:
            raise InputValidationError(
                f"At least 2 responses required, got {len(responses)}. "
                f"Recommended: 5 responses at temperature > 0.5."
            )

        cleaned=[r.strip() for r in responses if r and r.strip()]
        if len(cleaned)<2:
            raise InputValidationError(
                "At least 2 non-empty responses required after cleaning."
            )

        try:
            if self.fast_mode:
                return self._compute_embedding_dispersion(cleaned)
            else:
                return self._compute_entropy(cleaned)
        except (ComputationError, ModelLoadError):
            raise
        except Exception as e:
            raise ComputationError(f"Semantic entropy computation failed: {e}")