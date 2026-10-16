import re

from ..exceptions import BackendError, ConfigurationError

SCORE_PATTERN = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?|\.\d+)", re.IGNORECASE)

JUDGE_PROMPT = """You are checking whether an answer is faithful to its source context.

Context:
{context}

Question: {question}

Answer: {answer}

Step 1: List the factual claims the answer makes, including any direction (yes/no, increase/decrease), numbers, \
names and dates.

Step 2: Check each claim against the context. Is it explicitly supported, a reasonable inference, contradicted, \
or not mentioned?

Step 3: Score the whole answer using these anchors:
- 0.9-1.0: every claim is explicitly supported by the context
- 0.5-0.7: claims are reasonable inferences from the context but not explicitly stated
- 0.0-0.3: any claim contradicts the context or is not present in it

A flipped direction or a changed number is a contradiction and must score 0.0-0.3, even if the rest of the answer is \
correct.

Write Step 1 and Step 2 briefly, then on the final line write exactly:
SCORE: <a number between 0 and 1>
"""


class LLMJudge:
    def __init__(self, fn):
        if not callable(fn):
            raise ConfigurationError("judge must be a callable that takes a prompt and returns the model's text.")
        self.fn = fn

    def score(self, question, answer, chunks):
        context = "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
        prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
        try:
            text = str(self.fn(prompt)).strip()
        except Exception as e:
            raise BackendError(f"Judge call failed: {e}") from e
        matches = SCORE_PATTERN.findall(text)
        if not matches:
            return None, text
        score = float(matches[-1])
        if not 0.0 <= score <= 1.0:
            return None, text
        return score, SCORE_PATTERN.sub("", text).strip()
