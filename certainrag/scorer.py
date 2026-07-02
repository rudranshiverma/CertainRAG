from dataclasses import dataclass

@dataclass
class CertainRAGResult:
    question:str
    answer:str
    retrieval_confidence:float=None
    faithfulness:float=None
    faithfulness_reasoning:str=None
    self_consistency:float=None
    uncertainty:float=None

    def to_dict(self):
        return {
            "question":self.question,
            "answer":self.answer,
            "retrieval_confidence":self.retrieval_confidence,
            "faithfulness":self.faithfulness,
            "faithfulness_reasoning":self.faithfulness_reasoning,
            "self_consistency":self.self_consistency,
            "uncertainty":self.uncertainty,
        }

DEFAULT_WEIGHTS = {
    "retrieval_confidence": 0.3,
    "faithfulness": 0.5,
    "self_consistency": 0.2,
}
def combine_scores(
    retrieval_confidence=None,
    faithfulness=None,
    self_consistency=None,
    weights=None,
):
    weights=weights or DEFAULT_WEIGHTS
    confidences={}

    if retrieval_confidence is not None:
        confidences["retrieval_confidence"]=retrieval_confidence
    if faithfulness is not None:
        confidences["faithfulness"]=faithfulness
    if self_consistency is not None:
        # self_consistency is a disagreement score, so must be flipped to a confidence
        confidences["self_consistency"] = 1.0-self_consistency

    if not confidences:
        return None

    total_weight=sum(weights[key] for key in confidences)
    if total_weight==0:
        combined_confidence=sum(confidences.values())/len(confidences)
    else:
        combined_confidence=0.0
        for key in confidences:
            combined_confidence+=confidences[key]*weights[key]
        combined_confidence=combined_confidence/total_weight

    return 1.0-combined_confidence