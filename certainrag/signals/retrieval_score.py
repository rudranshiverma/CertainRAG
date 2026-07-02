from ..utils import rank_weights, weighted_mean

class RetrievalConfidenceSignal:
    def score(self, retrieval_scores):
        if not retrieval_scores:
            return None
        valid_scores=[]
        for s in retrieval_scores:
            if s is not None and isinstance(s, (int, float)) and s >=0:
                valid_scores.append(float(s))
        if not valid_scores:
            return None
        valid_scores.sort(reverse=True)
        weights=rank_weights(len(valid_scores))
        return weighted_mean(valid_scores, weights)