import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from .. import config
from ..embeddings.embedder import embed
from ..storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class RetrievalResult:
    """
    Wraps a retrieved chunk with its vector cosine score and reranker score.
    """
    def __init__(self, payload: Dict[str, Any], vector_score: float, reranker_score: Optional[float] = None):
        self.chunk_id: str = payload.get("chunk_id", "")
        self.doc_id: str = payload.get("doc_id", "")
        self.chunk_type: str = payload.get("chunk_type", "text")
        self.content: str = payload.get("content", "")
        self.page: int = payload.get("page", 0)
        self.atomic: bool = payload.get("atomic", False)
        self.image_path: Optional[str] = payload.get("image_path")
        self.parent_section: Optional[str] = payload.get("parent_section")
        self.vector_score: float = vector_score
        self.reranker_score: Optional[float] = reranker_score

    @property
    def score(self) -> float:
        """Return reranker_score if available, otherwise vector_score."""
        return self.reranker_score if self.reranker_score is not None else self.vector_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "page": self.page,
            "atomic": self.atomic,
            "image_path": self.image_path,
            "parent_section": self.parent_section,
            "vector_score": round(self.vector_score, 4),
            "reranker_score": round(self.reranker_score, 4) if self.reranker_score is not None else None,
            "score": round(self.score, 4),
        }


def retrieve(
    query: str,
    collection_name: Optional[str] = None,
    store: Optional[QdrantStore] = None,
    top_k: Optional[int] = None,
    doc_id_filter: Optional[str] = None,
    chunk_type_filter: Optional[str] = None,
) -> Tuple[List[RetrievalResult], Dict[str, float]]:
    """
    Embeds the query and retrieves the top-k most similar chunks from Qdrant.

    Returns
    -------
    tuple[list[RetrievalResult], dict[str, float]]
        (retrieved_candidates, timing_metrics_ms)
    """
    t_start = time.perf_counter()

    if top_k is None:
        top_k = config.TOP_K_RETRIEVAL

    # 1. Query Embedding
    t0 = time.perf_counter()
    query_vector = embed(query).tolist()
    query_embed_ms = (time.perf_counter() - t0) * 1000

    # 2. Qdrant Search
    if store is None:
        target_col = collection_name or config.COLLECTION_NAME
        store = QdrantStore(target_col)

    t0 = time.perf_counter()
    raw_results = store.search(
        query_vector=query_vector,
        top_k=top_k,
        doc_id_filter=doc_id_filter,
        chunk_type_filter=chunk_type_filter,
        score_threshold=0.0,
    )
    qdrant_search_ms = (time.perf_counter() - t0) * 1000

    # 3. Parse Results
    t0 = time.perf_counter()
    candidates = [RetrievalResult(hit.payload, hit.score) for hit in raw_results]
    result_parse_ms = (time.perf_counter() - t0) * 1000

    total_retrieval_ms = (time.perf_counter() - t_start) * 1000

    timings = {
        "query_embed_ms": round(query_embed_ms, 1),
        "qdrant_search_ms": round(qdrant_search_ms, 1),
        "result_parse_ms": round(result_parse_ms, 1),
        "total_retrieval_ms": round(total_retrieval_ms, 1),
    }

    logger.debug(
        "Retrieval Timing: embed=%.1fms, qdrant=%.1fms, parse=%.1fms, total=%.1fms",
        query_embed_ms, qdrant_search_ms, result_parse_ms, total_retrieval_ms,
    )

    return candidates, timings

