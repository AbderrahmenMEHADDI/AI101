import time
import logging
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

from .. import config

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton wrapper around SentenceTransformer.

    Responsible for converting text into dense vector embeddings.

    The model is loaded once per process.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            t0 = time.perf_counter()
            cls._instance = super().__new__(cls)

            model_name = config.EMBEDDING_MODEL
            target_device = getattr(config, "EMBEDDING_DEVICE", None)

            # Determine device & check CUDA availability
            if not target_device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = target_device

            cuda_available = torch.cuda.is_available()
            logger.info(
                "Initializing Embedding Model '%s' on device='%s' (CUDA available: %s)",
                model_name, device, cuda_available,
            )

            # Load model
            cls._instance.model = SentenceTransformer(
                model_name,
                device=device
            )
            cls._instance.model_name = model_name
            cls._instance.device = str(cls._instance.model.device)
            cls._instance.dimension = cls._instance.model.get_sentence_embedding_dimension()

            init_ms = (time.perf_counter() - t0) * 1000
            cls._instance.init_time_ms = round(init_ms, 1)

            logger.info(
                "Embedding model initialized in %.1f ms (device: %s, dim: %d)",
                init_ms, cls._instance.device, cls._instance.dimension,
            )

        return cls._instance

    # ---------------------------------------------------------
    # Single embedding
    # ---------------------------------------------------------

    def embed(
        self,
        text: str
    ) -> np.ndarray:
        """
        Embed a single piece of text.

        Returns a normalized float32 vector.
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot embed empty text."
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return vector.astype(
            np.float32
        )

    # ---------------------------------------------------------
    # Batch embedding
    # ---------------------------------------------------------

    def embed_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Embed multiple texts in a batch.

        Returns:
            shape = (number_of_texts, embedding_dimension)
        """

        if not texts:
            return np.empty(
                (
                    0,
                    self.dimension
                ),
                dtype=np.float32
            )

        cleaned_texts = [
            text.strip()
            for text in texts
        ]

        if any(
            not text
            for text in cleaned_texts
        ):
            raise ValueError(
                "embed_batch() received empty text."
            )

        vectors = self.model.encode(
            cleaned_texts,
            batch_size=getattr(
                config,
                "EMBEDDING_BATCH_SIZE",
                32
            ),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return vectors.astype(
            np.float32
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def get_dimension(self) -> int:
        """
        Return vector dimensionality.
        """

        return self.dimension

    def get_model_name(self) -> str:
        """
        Return embedding model name.
        """

        return self.model_name


# -------------------------------------------------------------
# Global instance
# -------------------------------------------------------------

embedder_instance = Embedder()


# -------------------------------------------------------------
# Convenience functions
# -------------------------------------------------------------

def embed(
    text: str
) -> np.ndarray:

    return embedder_instance.embed(
        text
    )


def embed_batch(
    texts: List[str]
) -> np.ndarray:

    return embedder_instance.embed_batch(
        texts
    )