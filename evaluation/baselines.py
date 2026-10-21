import torch

HHEM_CHECKPOINT = "vectara/hallucination_evaluation_model"
LETTUCEDETECT_CHECKPOINTS = {
    "lettucedetect-base": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "lettucedetect-large": "KRLabsOrg/lettucedect-large-modernbert-en-v1",
}


class HHEM:
    def __init__(self, device):
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(HHEM_CHECKPOINT, trust_remote_code=True)
        self.model.to(device).eval()

    def __call__(self, record):
        premise = "\n".join(record["chunks"])
        with torch.inference_mode():
            score = self.model.predict([(premise, record["answer"])])
        return {"score": float(score[0])}


class LettuceDetect:
    def __init__(self, name, device):
        from lettucedetect.models.inference import HallucinationDetector

        self.detector = HallucinationDetector(
            method="transformer", model_path=LETTUCEDETECT_CHECKPOINTS[name], device=torch.device(device)
        )

    def __call__(self, record):
        if "prompt" in record:
            tokens = self.detector.predict_prompt(record["prompt"], record["answer"], output_format="tokens")
        else:
            tokens = self.detector.predict(
                context=record["chunks"], answer=record["answer"], question=record["question"], output_format="tokens"
            )
        # token prob is P(hallucinated); the answer is as unsupported as its worst token
        return {"score": 1.0 - max((t["prob"] for t in tokens), default=0.0)}
