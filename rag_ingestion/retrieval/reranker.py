import logging
import time
from typing import List, Optional, Tuple
import torch
from sentence_transformers import CrossEncoder

from .. import config
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


class _CrossEncoderReranker:
    """
    Singleton wrapper around the CrossEncoder model.
    Loads the model once per process and reuses it across queries.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            t0 = time.perf_counter()
            cls._instance = super().__new__(cls)

            model_name = config.CROSS_ENCODER_MODEL
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"

            logger.info(
                "Initializing CrossEncoder model '%s' on device='%s' (CUDA available: %s)...",
                model_name, device, cuda_available,
            )

            try:
                cls._instance.model = CrossEncoder(model_name, device=device)
            except Exception as exc:
                if device == "cuda":
                    logger.warning("Failed to load CrossEncoder on CUDA (%s). Falling back to CPU.", exc)
                    device = "cpu"
                    cls._instance.model = CrossEncoder(model_name, device=device)
                else:
                    raise

            cls._instance.device = device
            init_ms = (time.perf_counter() - t0) * 1000
            cls._instance.init_time_ms = round(init_ms, 1)

            logger.info(
                "CrossEncoder initialized in %.1f ms (device: %s).",
                init_ms, device,
            )

        return cls._instance

    def score(self, query: str, passages: List[str]) -> Tuple[List[float], float]:
        """
        Score passages against a query and return (scores, inference_ms).
        """
        t0 = time.perf_counter()
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        inference_ms = (time.perf_counter() - t0) * 1000
        return scores.tolist(), inference_ms


def warmup_reranker() -> _CrossEncoderReranker:
    """Explicitly instantiate and warm up the CrossEncoder singleton."""
    return _CrossEncoderReranker()


def rerank(query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> Tuple[List[RetrievalResult], float]:
    """
    Re-scores candidate retrieval results with a cross-encoder.

    Returns
    -------
    tuple[list[RetrievalResult], float]
        (reranked_results, inference_ms)
    """
    if top_k is None:
        top_k = config.TOP_K_RERANK

    if not results:
        return [], 0.0

    if len(results) == 1:
        results[0].reranker_score = results[0].vector_score
        return results[:top_k], 0.0

    reranker = _CrossEncoderReranker()
    passages = [r.content for r in results]
    scores, inference_ms = reranker.score(query, passages)

    # Attach reranker_score to each result without losing vector_score
    for result, score in zip(results, scores):
        result.reranker_score = float(score)

    reranked = sorted(results, key=lambda r: r.reranker_score if r.reranker_score is not None else -999.0, reverse=True)
    return reranked[:top_k], round(inference_ms, 1)

