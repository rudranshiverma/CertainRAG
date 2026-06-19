import numpy as np
from certainrag.exceptions import InputValidationError, ComputationError

"""measures how relevant the retrieved chunks are to the query.
expects normalized cosine similarity scores in [0,1] from the developer's retriever.
if your retriever (such as FAISS) return raw L2 distance, convert first using certainrag.utils.normalize_l2()."""
class RetrievalConfidenceSignal:
    def compute(self, retrieval_scores:list[float])->dict:
        if not retrieval_scores:
            raise InputValidationError("retriever_scores cannot be empty. "
                                       "Pass atleast one similarity score from your retriever.")
        if not isinstance(retrieval_scores, (list,tuple)):
            raise InputValidationError(f"retrieval_scores must be a list, got {type(retrieval_scores)}")
        scores=[float(s) for s in retrieval_scores]
        out_of_range=[s for s in scores if not (0.0<=s<=1.0)]
        if out_of_range:
            raise InputValidationError(
                f"retrieval_scores must be normalized similarity scores in [0,1]. "
                f"Found invalid values: {out_of_range}. "
                f"If using FAISS L2 distance, normalize first with certainrag.utils.normalize_l2(). "
                f"If using LangChain FAISS, prefer similarity_search_with_relevance_scores()."
            )
        try:
            confidence=float(np.mean(scores))
        except Exception as e:
            raise ComputationError(f"Failed to compute retrieval confidence: {e}")
        return{
            "retrieval_confidence": confidence,
            "scores": scores
        }