"""
Generator
=========
Modular LLM generation layer with multiple backends.

Architecture
------------

    BaseGenerator          (abstract interface)
        │
        ├── OllamaGenerator    (default — free, local)
        └── GeminiGenerator    (optional — cloud, requires API key)

The backend is selected by ``config.GENERATOR_BACKEND``:
    "ollama"  →  OllamaGenerator
    "gemini"  →  GeminiGenerator

Usage
-----
    from rag_ingestion.generation.generator import get_generator
    gen = get_generator()
    result = gen.generate("Your full prompt here")
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .. import config
from ..retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class GenerationResult:
    """Structured container for generated text and telemetry metrics."""
    def __init__(
        self,
        text: str,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
        tokens_per_sec: float = 0.0,
    ):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.duration_ms = duration_ms
        self.tokens_per_sec = tokens_per_sec


class BaseGenerator(ABC):
    """Abstract interface for LLM text generation backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text response."""
        ...

    def generate_with_metrics(self, prompt: str) -> GenerationResult:
        """Generate text response and return structured GenerationResult with telemetry."""
        import time
        t0 = time.perf_counter()
        text = self.generate(prompt)
        duration_ms = (time.perf_counter() - t0) * 1000
        return GenerationResult(text=text, duration_ms=duration_ms)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


class OllamaGenerator(BaseGenerator):
    """
    Generate answers using a locally running Ollama model.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ):
        self._model = model or config.GENERATION_MODEL
        self._temperature = temperature if temperature is not None else config.TEMPERATURE
        self._max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        self._timeout = timeout or config.GENERATION_TIMEOUT
        self._enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"

    def generate(self, prompt: str) -> str:
        res = self.generate_with_metrics(prompt)
        return res.text

    def generate_with_metrics(self, prompt: str) -> GenerationResult:
        import time
        try:
            import ollama  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "ollama package not installed.\n"
                "Install with: pip install ollama\n"
                "Then: ollama pull " + self._model
            ) from exc

        logger.debug(
            "Generating with Ollama model='%s' temp=%.2f max_tokens=%d thinking=%s",
            self._model, self._temperature, self._max_tokens, self._enable_thinking,
        )

        options: Dict[str, Any] = {
            "temperature": self._temperature,
            "num_predict": self._max_tokens,
        }

        t0 = time.perf_counter()
        try:
            response = ollama.generate(
                model=self._model,
                prompt=prompt,
                options=options,
            )
            wall_ms = (time.perf_counter() - t0) * 1000
            answer = response.get("response", "").strip()

            # Clean up thinking tags if present in answer and thinking is disabled
            if not self._enable_thinking and "<think>" in answer:
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()

        except Exception as exc:
            raise RuntimeError(
                f"Ollama generation failed for model '{self._model}': {exc}\n"
                "Make sure Ollama is running (ollama serve) and the model is pulled."
            ) from exc

        if not answer:
            raise RuntimeError(
                f"Ollama returned an empty response (model='{self._model}')."
            )

        # Extract Ollama native telemetry
        prompt_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        eval_duration_ns = response.get("eval_duration", 0)

        if eval_duration_ns > 0 and output_tokens > 0:
            duration_sec = eval_duration_ns / 1e9
            tokens_per_sec = output_tokens / duration_sec
            duration_ms = duration_sec * 1000
        else:
            duration_ms = wall_ms
            tokens_per_sec = (output_tokens / (wall_ms / 1000)) if wall_ms > 0 else 0.0

        return GenerationResult(
            text=answer,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration_ms, 1),
            tokens_per_sec=round(tokens_per_sec, 2),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini backend (optional, cloud)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiGenerator(BaseGenerator):
    """
    Generate answers using the Google Gemini API.

    Requires GEMINI_API_KEY in .env.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._model_id = model or config.GEMINI_MODEL
        self._temperature = temperature if temperature is not None else config.TEMPERATURE
        self._max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        self._client = None

    @property
    def model_name(self) -> str:
        return f"gemini:{self._model_id}"

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai not installed.\n"
                "Install with: pip install google-generativeai"
            ) from exc

        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to .env or set the environment variable."
            )

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self._model_id)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        logger.debug("Generating with Gemini model='%s'", self._model_id)

        try:
            response = client.generate_content(
                prompt,
                generation_config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_tokens,
                },
            )
            answer = response.text.strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        if not answer:
            raise RuntimeError("Gemini returned an empty response.")

        return answer


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_generator() -> BaseGenerator:
    """
    Return the configured generator backend.

    Controlled by ``config.GENERATOR_BACKEND`` / ``GENERATOR_BACKEND`` env var.
    Valid values: "ollama" (default), "gemini".
    """
    backend = config.GENERATOR_BACKEND.lower().strip()

    if backend == "ollama":
        return OllamaGenerator()
    elif backend == "gemini":
        return GeminiGenerator()
    else:
        raise ValueError(
            f"Unknown GENERATOR_BACKEND='{backend}'. "
            "Valid options: 'ollama', 'gemini'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# High-level convenience function (used by API and RAGPipeline)
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: List[RetrievalResult],
) -> Dict[str, Any]:
    """
    Build a prompt from chunks and generate a grounded answer.

    This function is the backwards-compatible entry point used by ``api.py``.
    For new code, prefer using ``RAGPipeline.answer()`` directly.

    Returns
    -------
    dict
        {
            "answer"   : str,
            "sources"  : list[dict],
            "model"    : str,
            "grounded" : bool,
        }
    """
    from .context_builder import ContextBuilder
    from .prompt_builder import PromptBuilder

    sources = [c.to_dict() for c in chunks]

    context = ContextBuilder().build(chunks)
    prompt = PromptBuilder().build(context.text, question)

    try:
        gen = get_generator()
        answer = gen.generate(prompt.full_prompt)
        return {
            "answer": answer,
            "sources": sources,
            "model": gen.model_name,
            "grounded": True,
        }
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        # Provide context as fallback rather than returning nothing
        fallback = (
            f"⚠️ **Generation failed:** {exc}\n\n"
            "**Retrieved context (generation unavailable):**\n\n"
            + context.text
        )
        return {
            "answer": fallback,
            "sources": sources,
            "model": "fallback",
            "grounded": False,
        }
