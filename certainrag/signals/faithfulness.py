import re
import numpy as np
from certainrag.exceptions import (InputValidationError, ModelLoadError, ComputationError)

#measures if the generated answer is supported by the retrieved context using NLI model DeBERTa
class FaithfulnessSignal:
    def __init__(self, model_name:str="cross-encoder/nli-deberta-v3-base"):
        self.model_name=model_name
        self._pipeline=None
    def get_pipeline(self):         #exposes loaded nli pipeline so semantic_entropy.py can reuse it without loading deberta twice
        self._load_model()
        return self._pipeline
    def _load_model(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline=pipeline("text-classification",model=self.model_name,top_k=None)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load NLI model '{self.model_name}'. "
                f"Ensure transformers is installed and model is accessible. "
                f"Error: {e}"
            )
    def _split_sentences(self,text:str)->list[str]:
        sentences=re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip())>10]
    def _entailment_prob(self,premise:str,hypothesis:str)->float:
        try:
            result=self._pipeline(f"{premise} [SEP] {hypothesis}")[0]
            for item in result:
                if item["label"].lower()=="entailment":
                    return float(item["score"])
            return 0.0
        except Exception as e:
            raise ComputationError(f"NLI inference failed: {e}")
    def compute(self,answer:str,context_chunks:list[str],retrieval_scores:list[float])->dict:
        if not answer or not answer.strip():
            raise InputValidationError("answer cannot be empty.")
        if not context_chunks:
            raise InputValidationError("context_chunks cannot be empty")
        if len(context_chunks)!=len(retrieval_scores):
            raise InputValidationError(
                f"context_chunks length ({len(context_chunks)}) must match retrieval_scores length ({len(retrieval_scores)})."
            )
        scores=[float(s) for s in retrieval_scores]
        if sum(scores)==0:
            raise InputValidationError("All retrieval_scores are zero. Cannot compute weighted faithfulness")
        self._load_model()
        sentences=self._split_sentences(answer)
        if not sentences:
            sentences=[answer.strip()]
        chunk_scores=[]
        for chunk,ret_score in zip(context_chunks,scores):
            if not chunk or not chunk.strip():
                continue
            sentence_probs=[self._entailment_prob(chunk,sentence) for sentence in sentences]
            chunk_entailment=float(np.mean(sentence_probs))
            chunk_scores.append({
                "chunk":chunk,
                "entailment_prob":chunk_entailment,
                "retrieval_score":ret_score,
                "sentence_scores":sentence_probs
            })
        if not chunk_scores:
            raise ComputationError("No valid chunks to score.")
        
        weights=np.array([c["retrieval_score"] for c in chunk_scores])
        probs=np.array([c["entailment_prob"] for c in chunk_scores])
        try:
            weighted_score=float(np.average(probs,weights=weights))
        except ZeroDivisionError:
            raise ComputationError("Weighted average failed. Weights sum to zero.")
        return{
            "faithfulness_score": weighted_score,
            "chunk_scores": chunk_scores,
            "supporting_chunks": [c for c in chunk_scores if c["entailment_prob"]>0.5],
            "contradicting_chunks": [c for c in chunk_scores if c["entailment_prob"]<0.2],
            "sentences_evaluated": sentences
        }