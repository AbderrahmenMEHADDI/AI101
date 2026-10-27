"""
RAG Pipeline
============
High-level orchestration of the full query → answer pipeline.

Flow
----
    query
      ↓
    embed query
      ↓
    Qdrant retrieval  (top TOP_K_RETRIEVAL candidates)
      ↓
    CrossEncoder reranking  (keep FINAL_TOP_K)
      ↓
    ContextBuilder  (token-budgeted context)
      ↓
    PromptBuilder   (system + context + question)
      ↓
    Generator       (Ollama / Gemini)
      ↓
    answer + sources + debug info

Usage
-----
    from rag_ingestion.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(collection_name="my_docs")
    result = pipeline.answer("What is the main contribution?")
    print(result["answer"])
    print(result["sources"])
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from . import config
from .embeddings.embedder import embedder_instance
from .storage.qdrant_store import QdrantStore
from .retrieval.retriever import retrieve, RetrievalResult
from .retrieval.reranker import rerank
from .generation.context_builder import ContextBuilder, BuiltContext
from .generation.prompt_builder import PromptBuilder
from .generation.generator import get_generator, BaseGenerator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceInfo:
    """Metadata about one retrieved+ranked source."""
    doc_id: str
    page: int
    chunk_id: str
    chunk_type: str
    score: float
    content: str
    image_path: Optional[str] = None


@dataclass
class RAGResult:
    """Full structured result from a RAG query."""
    answer: str
    sources: List[SourceInfo] = field(default_factory=list)
    model: str = ""
    grounded: bool = False
    latency_ms: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG query pipeline.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        enable_reranking: bool = True,
        generator: Optional[BaseGenerator] = None,
        auto_warmup: bool = False,
    ):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.top_k_retrieval = top_k_retrieval or config.TOP_K_RETRIEVAL
        self.top_k_rerank = top_k_rerank or config.FINAL_TOP_K
        self.enable_reranking = enable_reranking

        self._context_builder = ContextBuilder()
        self._prompt_builder = PromptBuilder()
        self._generator = generator or get_generator()
        self._store: Optional[QdrantStore] = None

        if auto_warmup or getattr(config, "ENABLE_WARMUP", False):
            self.warmup()

    @property
    def store(self) -> QdrantStore:
        """Lazy-loaded or pre-warmed Qdrant store instance."""
        if self._store is None:
            self._store = QdrantStore(self.collection_name)
        return self._store

    def warmup(self, dummy_inference: bool = True) -> Dict[str, float]:
        """
        Explicitly pre-load all models and validate Qdrant connection/schema.
        """
        logger.info("==================================================")
        logger.info("WARMING UP RAG PIPELINE COMPONENTS")
        logger.info("==================================================")
        t_start = time.perf_counter()

        # 1. Warm up Embedder
        from .embeddings.embedder import embedder_instance
        emb_init = embedder_instance.init_time_ms

        # 2. Warm up Reranker
        if self.enable_reranking:
            from .retrieval.reranker import warmup_reranker
            rr = warmup_reranker()
            rr_init = rr.init_time_ms
        else:
            rr_init = 0.0

        # 3. Warm up Qdrant Store (schema validation)
        t0 = time.perf_counter()
        _ = self.store
        qdrant_init = (time.perf_counter() - t0) * 1000

        # 4. Optional dummy inference
        dummy_ms = 0.0
        if dummy_inference:
            t0 = time.perf_counter()
            try:
                _ = embedder_instance.embed("warmup query")
                if self.enable_reranking:
                    from .retrieval.reranker import rerank
                    _ = rerank("warmup query", [RetrievalResult({"content": "warmup text"}, vector_score=1.0)])
            except Exception as exc:
                logger.warning("Warmup dummy inference notice: %s", exc)
            dummy_ms = (time.perf_counter() - t0) * 1000

        total_warmup = (time.perf_counter() - t_start) * 1000

        logger.info("Embedder init:     %.1f ms", emb_init)
        if self.enable_reranking:
            logger.info("Reranker init:     %.1f ms", rr_init)
        logger.info("Qdrant store init: %.1f ms", qdrant_init)
        logger.info("Total warmup time: %.1f ms", total_warmup)
        logger.info("==================================================")

        return {
            "embedder_init_ms": emb_init,
            "reranker_init_ms": rr_init,
            "qdrant_init_ms": round(qdrant_init, 1),
            "dummy_inference_ms": round(dummy_ms, 1),
            "total_warmup_ms": round(total_warmup, 1),
        }

    def answer(
        self,
        query: str,
        doc_id_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> RAGResult:
        """
        Answer a natural-language question using the RAG pipeline.
        """
        t_start = time.perf_counter()
        query = query.strip()

        if not query:
            return RAGResult(
                answer="Please provide a non-empty question.",
                grounded=False,
            )

        debug: Dict[str, Any] = {"query": query}

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        logger.info("Query: %s", query)

        candidates, r_timings = retrieve(
            query=query,
            store=self.store,
            top_k=self.top_k_retrieval,
            doc_id_filter=doc_id_filter,
            chunk_type_filter=chunk_type_filter,
        )

        debug["retrieval_candidates"] = len(candidates)
        debug["query_embed_ms"] = r_timings["query_embed_ms"]
        debug["qdrant_search_ms"] = r_timings["qdrant_search_ms"]
        debug["result_parse_ms"] = r_timings["result_parse_ms"]
        debug["retrieval_ms"] = r_timings["total_retrieval_ms"]

        logger.info(
            "Retrieved %d candidates in %.1f ms (qdrant_search: %.1f ms)",
            len(candidates), r_timings["total_retrieval_ms"], r_timings["qdrant_search_ms"],
        )

        if not candidates:
            return RAGResult(
                answer="No relevant content was found in the knowledge base for your question.",
                grounded=False,
                latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
                debug=debug,
            )

        # ── Step 2: Rerank ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self.enable_reranking:
            try:
                ranked, rerank_ms = rerank(
                    query=query,
                    results=candidates,
                    top_k=self.top_k_rerank,
                )
                logger.info(
                    "Reranked → %d results in %.1f ms",
                    len(ranked), rerank_ms,
                )
                debug["rerank_ms"] = round(rerank_ms, 1)
            except Exception as exc:
                logger.warning("Reranking failed (%s); using retrieval order.", exc)
                ranked = sorted(candidates, key=lambda r: r.vector_score, reverse=True)[: self.top_k_rerank]
                debug["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            ranked = sorted(candidates, key=lambda r: r.vector_score, reverse=True)[: self.top_k_rerank]
            debug["rerank_ms"] = 0.0
            logger.info("Reranking disabled; using top-%d by retrieval score.", len(ranked))

        debug["final_chunks"] = len(ranked)

        # ── Step 3: Build context ─────────────────────────────────────────────
        t0 = time.perf_counter()
        context: BuiltContext = self._context_builder.build(ranked)
        context_ms = (time.perf_counter() - t0) * 1000

        debug["context_ms"] = round(context_ms, 1)
        debug["context_tokens"] = context.total_tokens
        debug["context_budget"] = context.budget_tokens
        debug["context_dropped"] = context.dropped_count

        logger.info(
            "Context built in %.1f ms: %d tokens / %d budget (%d chunks dropped)",
            context_ms, context.total_tokens, context.budget_tokens, context.dropped_count,
        )

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        t0 = time.perf_counter()
        prompt = self._prompt_builder.build(context.text, query)
        prompt_ms = (time.perf_counter() - t0) * 1000

        debug["prompt_ms"] = round(prompt_ms, 1)
        debug["prompt_tokens"] = prompt.total_tokens
        debug["model"] = self._generator.model_name

        # ── Step 5: Generate ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        grounded = True
        try:
            gen_res = self._generator.generate_with_metrics(prompt.full_prompt)
            answer_text = gen_res.text
            gen_ms = gen_res.duration_ms or ((time.perf_counter() - t0) * 1000)

            logger.info(
                "Generated answer in %.1f ms (model=%s, %d tokens, %.1f tok/s)",
                gen_ms, self._generator.model_name, gen_res.output_tokens, gen_res.tokens_per_sec,
            )
            debug["generation_ms"] = round(gen_ms, 1)
            debug["output_tokens"] = gen_res.output_tokens
            debug["tokens_per_sec"] = gen_res.tokens_per_sec
        except Exception as exc:
            logger.error("Generation error: %s", exc)
            answer_text = (
                f"⚠️ Generation failed: {exc}\n\n"
                "**Retrieved context:**\n\n" + context.text
            )
            grounded = False
            debug["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            debug["output_tokens"] = 0
            debug["tokens_per_sec"] = 0.0

        # ── Assemble result ───────────────────────────────────────────────────
        sources = [
            SourceInfo(
                doc_id=src.doc_id,
                page=src.page,
                chunk_id=src.chunk_id,
                chunk_type=src.chunk_type,
                score=src.score,
                content=src.content,
                image_path=src.image_path,
            )
            for src in context.sources
        ]

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        debug["total_ms"] = total_ms

        return RAGResult(
            answer=answer_text,
            sources=sources,
            model=self._generator.model_name,
            grounded=grounded,
            latency_ms=total_ms,
            debug=debug,
        )
