from abc import ABC, abstractmethod
import requests
from .exceptions import BackendError

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt, temperature=0.0, n=1):
        """
        Generate n responses for the given prompt.
        Returns a list of strings of length n.
        temperature=0.0 by default for deterministic output.
        Pass a higher temperature explicitly when variation is needed
        (e.g. SelfConsistencySignal uses temperature=1.0).
        """
        ...

class OllamaClient(LLMClient):
    def __init__(self, model="mistral", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt, temperature=0.0, n=1):
        outputs = []
        for _ in range(n):
            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=120,
                )
                response.raise_for_status()
                outputs.append(response.json().get("response", "").strip())
            except Exception as e:
                raise BackendError(f"Ollama generate failed: {e}") from e
        return outputs