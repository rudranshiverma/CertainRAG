from dataclasses import dataclass
from certainrag.exceptions import InputValidationError, ComputationError

@dataclass
class UncertaintyResult:
    uncertainty_score:float
    uncertainty_level:str       # can be low, medium or high
    should_abstain:bool
    retrieval_confidence:float
    faithfulness_score:float
    semantic_entropy:float
    signal_breakdown:dict
    supporting_chunks:list
    contradicting_chunks:list
    explanation:str
    latency_ms:float

class CompositeScorer:
    def __init__(self, weights:tuple=(0.33,0.33,0.34), threshold:float=0.5, low_threshold:float=0.35,high_threshold:float=0.65):
        if len(weights)!=3:
            raise InputValidationError("weights must be a tuple of 3 values: (retrieval_weight, faithfulness_weight, entropy_weight)")
        if not (0.99<=sum(weights)<=1.01):
            raise InputValidationError(f"weights must sum to 1.0, got {sum(weights):.3f}")
        if not 0.0<=threshold<=1.0:
            raise InputValidationError(f"threshold must be between 0 and 1, got {threshold}")
        self.w_retrieval,self.w_faith,self.w_entropy=weights
        self.threshold=threshold
        self.low_threshold=low_threshold
        self.high_threshold=high_threshold
    def _clip_float_drift(self, value: float, name: str) -> float:
        #Corrects floating point precision drift while still catching genuine out-of-range errors.
        value=float(value)
        epsilon=1e-9
        if -epsilon<=value<0.0:
            return 0.0
        if 1.0<value<=1.0+epsilon:
            return 1.0
        if not 0.0<=value<=1.0:
            raise InputValidationError(f"{name} must be between 0 and 1, got {value}")
        return value
    def score(self,
              retrieval_confidence:float,faithfulness_score:float,semantic_entropy:float,
              supporting_chunks:list, contradicting_chunks:list, latency_ms:float)->UncertaintyResult:
        
        # higher uncertainty happens with low retrieval confidence, low faithfulness and high entropy
        retrieval_confidence=self._clip_float_drift(retrieval_confidence, "retrieval_confidence")
        faithfulness_score=self._clip_float_drift(faithfulness_score, "faithfulness_score")
        semantic_entropy=self._clip_float_drift(semantic_entropy, "semantic_entropy")
        try:
            uncertainty=round(float(
                self.w_retrieval*(1-retrieval_confidence)+
            self.w_faith*(1-faithfulness_score)+
            self.w_entropy*semantic_entropy),4)
        except Exception as e:
            raise ComputationError(f"Composite score computation failed: {e}")
        if uncertainty<self.low_threshold:
            level="LOW"
        elif uncertainty<self.high_threshold:
            level="MEDIUM"
        else:
            level="HIGH"
        should_abstain=uncertainty>=self.threshold
        explanation=self._explain(retrieval_confidence,faithfulness_score,semantic_entropy,contradicting_chunks)
        return UncertaintyResult(
            uncertainty_score=uncertainty,
            uncertainty_level=level,
            should_abstain=should_abstain,
            retrieval_confidence=round(retrieval_confidence,4),
            faithfulness_score=round(faithfulness_score,4),
            semantic_entropy=round(semantic_entropy,4),
            signal_breakdown={
                "retrieval_confidence":retrieval_confidence,
                "faithfulness_score":faithfulness_score,
                "semantic_entropy":semantic_entropy,
                "weights":{
                    "retrieval":self.w_retrieval,
                    "faithfulness":self.w_faith,
                    "entropy":self.w_entropy
                }
            },
            supporting_chunks=supporting_chunks,
            contradicting_chunks=contradicting_chunks,
            explanation=explanation,
            latency_ms=round(float(latency_ms),2)
        )
    def _explain(self,ret,faith,entropy,contradicting):
        parts=[]
        if ret<0.4:
            parts.append("Retrieved context was weakly relevant")
        if faith<0.4:
            parts.append(f"Answer contradicted by {len(contradicting)} chunk(s)")
        if entropy>0.6:
            parts.append("Model gave inconsistent answers across samples")
        if not parts:
            return "Answer appears well supported by retrieved context"
        return "Flagged because: " + ", ".join(parts)