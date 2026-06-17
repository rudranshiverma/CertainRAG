from transformers import pipeline
import numpy as np
import re

class FaithfulnessScorer:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        self.nli=pipeline("text-classification", model=model_name, top_k=None)
    
    def _split_into_sentences(self,text:str)->list[str]:
        sentences=re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s) > 10]

    def _get_entailment_prob(self,premise:str, hypothesis:str)->float:
        result=self.nli(f"{premise} [SEP] {hypothesis}")[0]
        for item in result:
            if item["label"].lower()=="entailment":
                return item["score"]
        return 0.0

    def score(self, answer:str, chunks:list[dict])->dict:
        #chunks=list of {"chunks":str, "similarity_score":float}
        sentences=self._split_into_sentences(answer)
        if not sentences:
            sentences = [answer]
        chunk_scores=[]
        for item in chunks:
            premise=item["chunk"]
            retrieval_score=item["similarity_score"]
            sentence_probs=[
                self._get_entailment_prob(premise,sentence)
                for sentence in sentences
            ]
            chunk_entailment=float(np.mean(sentence_probs))
            chunk_scores.append({
                "chunk":premise,
                "entailment_prob":chunk_entailment,
                "retrieval_score":retrieval_score,
                "sentence_scores":sentence_probs
            })
        
        weights=np.array([c["retrieval_score"] for c in chunk_scores])
        probs=np.array([c["entailment_prob"] for c in chunk_scores])
        weighted_score=float(np.average(probs,weights=weights))

        supporting=[c for c in chunk_scores if c["entailment_prob"]>0.5]
        contradicting=[c for c in chunk_scores if c["entailment_prob"]<0.3]

        return{
            "faithfulness_score":weighted_score,
            "chunk_scores":chunk_scores,
            "supporting_chunks":supporting,
            "contradicting_chunks":contradicting,
            "sentences_evaluated": sentences
        }