import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..calibration import apply_platt
from ..result import RetrievalResult
from ..utils import pack_windows, split_sentences

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_PLATT = (1.0, 0.0)
MIN_WINDOW_TOKENS = 64


class RetrievalSignal:
    def __init__(self, model=DEFAULT_RERANKER, device=None, platt=DEFAULT_PLATT, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.platt = platt
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device).eval()
        self.max_length = min(self.tokenizer.model_max_length, 512)
        self._pair_overhead = self.tokenizer.num_special_tokens_to_add(pair=True)

    def logits(self, question, chunks):
        q_len = len(self.tokenizer(question, add_special_tokens=False, verbose=False)["input_ids"])
        budget = max(MIN_WINDOW_TOKENS, self.max_length - q_len - self._pair_overhead)
        owners, windows = [], []
        for i, chunk in enumerate(chunks):
            n = len(self.tokenizer(chunk, add_special_tokens=False, verbose=False)["input_ids"])
            pieces = [chunk] if n <= budget else pack_windows(split_sentences(chunk), self.tokenizer, budget) or [chunk]
            owners.extend([i] * len(pieces))
            windows.extend(pieces)
        out = np.full(len(chunks), -np.inf)
        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start : start + self.batch_size]
            batch = self.tokenizer(
                [question] * len(batch_windows),
                batch_windows,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits[:, 0].float().cpu().numpy()
            np.maximum.at(out, owners[start : start + self.batch_size], logits)
        return out

    def score(self, question, chunks):
        if not chunks:
            return RetrievalResult(score=0.0, chunk_scores=[])
        probs = apply_platt(self.logits(question, chunks), *self.platt)
        return RetrievalResult(score=float(probs.max()), chunk_scores=[float(p) for p in probs])
