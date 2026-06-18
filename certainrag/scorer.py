from dataclasses import dataclass
from typing import Optional

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
    def __init__(self, weights=(0.33,0.33,0.34), threshold=0.5):
        self.w_retrieval,self.w_faith,self.w_entropy=weights
        self.threshold=threshold
    def score(self,
              retrieval_confidence:float,faithfulness_score:float,semantic_entropy:float,
              supporting_chunks:list, contradicting_chunks:list, latency_ms:float)->UncertaintyResult:
        
        # higher uncertainty happens with low retrieval confidence, low faithfulness and high entropy
        uncertainty=(
            self.w_retrieval*(1-retrieval_confidence)+
            self.w_faith*(1-faithfulness_score)+
            self.w_entropy*semantic_entropy
        )
        if uncertainty<0.35:
            level="LOW"
        elif uncertainty<0.65:
            level="MEDIUM"
        else:
            level="HIGH"
        should_abstain=uncertainty>=self.threshold
        explanation=self._generate_explanation(retrieval_confidence,faithfulness_score,semantic_entropy,supporting_chunks,contradicting_chunks)
        return UncertaintyResult(
            uncertainty_score=round(uncertainty,4),
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
            latency_ms=latency_ms
        )
    def _generate_explanation(self,ret,faith,entropy,supporting,contradicting):
        parts=[]
        if ret<0.4:
            parts.append("Retrieved context was weakly relevant")
        if faith<0.4:
            parts.append(f"Answer contradicted by {len(contradicting)} chunk(s)")
        if entropy>0.6:
            parts.append("Model gave inconsistent answers across samples")
        if not parts:
            return "Answer appears well supported by retrieved context"
        return "Flagged because: " + " ".join(parts)