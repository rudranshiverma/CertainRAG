import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..exceptions import ConfigurationError

FACTCG_TEMPLATE = (
    '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "{text_b}"?\n\n'
    "OPTIONS:\n- Yes\n- No\nI think the answer is "
)
VERIFIERS = {
    "factcg": ("yaxili96/FactCG-DeBERTa-v3-Large", "factcg", 2048),
    "minicheck-deberta": ("lytang/MiniCheck-DeBERTa-v3-Large", "minicheck", 2048),
    "minicheck-roberta": ("lytang/MiniCheck-RoBERTa-Large", "minicheck", 512),
}
WINDOW_TOKENS = 400
MIN_WINDOW_TOKENS = 64
# headroom for tokenization differences at the window/claim boundary
MARGIN_TOKENS = 8


class Verifier:
    def __init__(self, name="factcg", device=None, batch_size=16):
        if name not in VERIFIERS:
            raise ConfigurationError(f"Unknown verifier '{name}'. Choose from: {', '.join(VERIFIERS)}")
        checkpoint, style, self.max_length = VERIFIERS[name]
        self.name = name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        self.model.to(self.device).eval()
        if style == "factcg":
            # FactCG was trained with eos as the padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self._format = lambda window, claim: FACTCG_TEMPLATE.format(text_a=window, text_b=claim)
        else:
            eos = self.tokenizer.eos_token
            self._format = lambda window, claim: window + eos + claim
        self._overhead = len(self.tokenizer(self._format("", ""))["input_ids"]) + MARGIN_TOKENS

    def prepare_claims(self, claims):
        limit = self.max_length - self._overhead - MIN_WINDOW_TOKENS
        ids = self.tokenizer(claims, add_special_tokens=False)["input_ids"]
        return [c if len(i) <= limit else self.tokenizer.decode(i[:limit]) for c, i in zip(claims, ids)]

    def window_budget(self, claims):
        longest = max(len(i) for i in self.tokenizer(claims, add_special_tokens=False)["input_ids"])
        return min(WINDOW_TOKENS, self.max_length - self._overhead - longest)

    def score(self, windows, claims):
        texts = [self._format(w, c) for w, c in zip(windows, claims)]
        probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].tolist())
        return probs
