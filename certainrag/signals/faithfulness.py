import re
from ..exceptions import BackendError, ConfigurationError

SCORE_PATTERN = re.compile(r"SCORE:\s*(\d*\.?\d+)", re.IGNORECASE)

JUDGE_PROMPT = """You are checking whether an answer is faithful to its source context.

Context:
{context}

Question: {question}

Answer: {answer}

Step 1: State the answer's core claim in one sentence (its yes/no/maybe direction and the specific facts/numbers it relies on).

Step 2: Check that core claim directly against the context. Does the context support that exact direction and those exact facts, or does it support the \
opposite, a different number, or something not mentioned at all?

Step 3: Score using these anchors, based only on what Step 2 found:
- 0.9-1.0: the core claim and its specific facts are explicitly supported by the context
- 0.5-0.7: the core claim is a reasonable inference, but not explicitly stated
- 0.0-0.3: the core claim, its direction, or a specific fact CONTRADICTS the context, or is fabricated and not present in the context at all

Do not default to a middle score. A flipped direction (yes vs no, increases vs decreases) or a changed number is a contradiction and must score 0.0-0.3, even \
if the rest of the answer reads fluently.

Write Step 1 and Step 2 briefly, then on the final line write exactly:
SCORE: <a number between 0 and 1>
"""

class FaithfulnessSignal:
    def __init__(self, llm_client):
        if llm_client is None:
            raise ConfigurationError("FaithfulnessSignal requires an llm_client.")
        self.llm_client=llm_client
    def score(self,question,answer,chunks):
        if not chunks:
            return 0.0, "No chunks were retrieved."
        context=""
        for i, chunk in enumerate(chunks):
            context+=f"[{i + 1}] {chunk}\n"
        prompt=JUDGE_PROMPT.format(context=context, question=question, answer=answer)
        try:
            outputs=self.llm_client.generate(prompt, temperature=0.0, n=1)
        except Exception as e:
            raise BackendError(f"Faithfulness judge call failed:{e}") from e
        text=outputs[0].strip() if outputs else ""
        match=SCORE_PATTERN.search(text)
        if not match:
            return 0.0, text
        score=float(match.group(1))
        score=max(0.0,min(1.0,score))
        reasoning=SCORE_PATTERN.sub("", text).strip()
        return score, reasoning