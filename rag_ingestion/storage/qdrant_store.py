"""
Qdrant Vector Store
===================
Handles all Qdrant interactions: collection management, vector upsert,
similarity search, and deletion.

Design principles
-----------------
- Embedding is NOT done here.  The store accepts pre-computed vectors.
  This keeps the storage layer decoupled from the embedding model.
- Vector dimension is derived from the Embedder singleton at runtime,
  not from a hard-coded config value.
- Distance metric: COSINE (works correctly with normalised unit vectors).
- IDs: UUID strings (Qdrant supports string UUIDs natively).
- Payload stores the full Chunk fields so retrieval returns rich metadata.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from .. import config
from ..schema import Chunk

logger = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    """Create a Qdrant client from config."""
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
    )


class QdrantStore:
    """
    Thin wrapper around the Qdrant client that encapsulates collection
    management, upsert, and search.

    Embedding is performed *outside* this class (see ``embedder.py``).
    The store only deals with raw vectors and metadata.
    """

    def __init__(self, collection_name: str):
        self.client = _get_client()
        self.collection_name = collection_name
        self._vector_size: Optional[int] = None
        self._vector_name: Optional[str] = None
        self._initialized: bool = False
        self._ensure_collection()

    # ─────────────────────────────────────────────────────────────────────────
    # Collection management
    # ─────────────────────────────────────────────────────────────────────────

    def _get_vector_size(self) -> int:
        """
        Derive the vector dimension from the Embedder singleton.

        This is intentionally lazy so that the store can be instantiated
        without loading the model (e.g. during tests).
        """
        if self._vector_size is None:
            from ..embeddings.embedder import embedder_instance
            self._vector_size = embedder_instance.get_dimension()
        return self._vector_size

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist, and detect its schema."""
        if self._initialized:
            return

        vector_size = self._get_vector_size()

        # Inspect existing collection to see if it uses named vectors
        # We use a raw HTTP request via httpx to bypass pydantic validation errors
        # that occur when qdrant-client versions lag behind the Qdrant Cloud API schema.
        import httpx

        headers = {}
        if config.QDRANT_API_KEY:
            headers["api-key"] = config.QDRANT_API_KEY

        url = f"{config.QDRANT_URL.rstrip('/')}/collections/{self.collection_name}"

        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                vectors_config = data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})

                if "size" not in vectors_config and vectors_config:
                    # Named vectors are in use (since 'size' is not a top-level key)
                    self._vector_name = list(vectors_config.keys())[0]
                    target_size = vectors_config[self._vector_name].get("size", vector_size)
                    logger.info(
                        "Detected named vector '%s' (dim=%d) in collection '%s'",
                        self._vector_name, target_size, self.collection_name
                    )
                else:
                    self._vector_name = None
                    target_size = vectors_config.get("size", vector_size)
                    logger.info(
                        "Detected unnamed vector (dim=%d) in collection '%s'",
                        target_size, self.collection_name
                    )

                if target_size != vector_size:
                    raise ValueError(
                        f"\nCRITICAL DIMENSION MISMATCH:\n"
                        f"Your Qdrant collection '{self.collection_name}' expects vectors of dimension {target_size}.\n"
                        f"But your currently configured embedding model ('{config.EMBEDDING_MODEL}') outputs dimension {vector_size}.\n\n"
                        f"To fix this, you must either:\n"
                        f"  1. Delete the collection in Qdrant and let the pipeline recreate it at dim {vector_size}\n"
                        f"  2. Change EMBEDDING_MODEL in your .env to a model that outputs {target_size} dimensions."
                    )
            elif resp.status_code == 404:
                logger.info(
                    "Collection '%s' does not exist. Creating it (dim=%d, distance=COSINE)...",
                    self.collection_name, vector_size,
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                self._vector_name = None
            else:
                logger.warning("Failed to inspect collection schema via HTTP (%d).", resp.status_code)
                self._vector_name = None
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning("HTTP schema inspection failed: %s", e)
            self._vector_name = None

        self._initialized = True

    def collection_exists(self) -> bool:
        """Return True if the collection is present in Qdrant."""
        cols = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in cols)

    def delete_collection(self) -> None:
        """Delete the collection (irreversible)."""
        self.client.delete_collection(self.collection_name)
        self._initialized = False
        logger.info("Deleted collection '%s'.", self.collection_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Upsert
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """
        Upsert chunks with their pre-computed embedding vectors.
        """
        if not chunks:
            return

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors."
            )

        expected_dim = self._get_vector_size()
        actual_dim = vectors.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}."
            )

        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type,
                "page": chunk.page,
                "content": chunk.content,
                "atomic": chunk.atomic,
                "token_count": chunk.token_count,
                "image_path": chunk.image_path,
                "parent_section": chunk.parent_section,
                "metadata": chunk.metadata,
            }

            formatted_vector = vector.tolist()
            if self._vector_name:
                formatted_vector = {self._vector_name: formatted_vector}

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=formatted_vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d points into '%s'.", len(points), self.collection_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[ScoredPoint]:
        """
        Run a cosine-similarity search.
        """
        conditions = []
        if doc_id_filter:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
            )
        if chunk_type_filter:
            conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter))
            )

        query_filter = Filter(must=conditions) if conditions else None

        if self._vector_name:
            q_vector = NamedVector(name=self._vector_name, vector=query_vector)
        else:
            q_vector = query_vector

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )
