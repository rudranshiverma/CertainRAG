from abc import ABC, abstractmethod
import requests
from .exceptions import BackendError, ConfigurationError

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
    def __init__(self, model="mistral",host="http://localhost:11434"):
        self.model=model
        self.host=host.rstrip("/")
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

class AnthropicClient(LLMClient):
    """LLMClient backed by the Anthropic messages API.
    Requires the `anthropic` package (pip install anthropic).
    """
    def __init__(self, model="claude-haiku-4-5-20251001", api_key=None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ConfigurationError(
                "The anthropic package is required for AnthropicClient. "
                "Install with: pip install anthropic"
            ) from e
        import os
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "No Anthropic API key found. Pass api_key= explicitly or set "
                "the ANTHROPIC_API_KEY environment variable."
            )
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt, temperature=0.0, n=1):
        # The Anthropic API doesn't support n>1 in a single call, so this loops n times. 
        # Relevant when this client is used for the self-consistency signal, which calls generate(n=3) by default.
        outputs=[]
        try:
            for _ in range(n):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                outputs.append(text.strip())
        except Exception as e:
            raise BackendError(f"Anthropic generate failed: {e}") from e
        return outputs


class OpenAICompatibleClient(LLMClient):
    """LLMClient for OpenAI and any provider that speaks the same chat
    completions protocol (Together, Groq, Fireworks, Mistral, DeepSeek,
    OpenRouter, Ollama's own OpenAI-compatible endpoint, etc.). just
    point base_url at the provider and pass its model name.
    Requires the `openai` package (pip install openai), which is used
    purely as an HTTP client here; it works against any compatible API,
    not just OpenAI's own.
    """

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ConfigurationError(
                "The openai package is required for OpenAICompatibleClient. "
                "Install with: pip install openai"
            ) from e
        import os
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "No API key found. Pass api_key= explicitly or set the "
                "OPENAI_API_KEY environment variable (or the equivalent "
                "for whichever OpenAI-compatible provider you're pointing at)."
            )
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, temperature=0.0, n=1):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            return [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            raise BackendError(f"OpenAI-compatible generate failed: {e}") from e