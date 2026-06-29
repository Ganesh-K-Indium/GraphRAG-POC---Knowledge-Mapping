"""
Qdrant-backed vector store with hybrid dense+sparse search.

Accepts a per-workspace collection name.

Design notes
------------
- The collection is created automatically on first instantiation if it does
  not already exist (``_ensure_collection``).
- Supports hybrid search: dense vectors (BGE-M3) + sparse BM25 vectors.
  The collection uses named vectors: ``"dense"`` for semantic similarity
  and ``"bm25"`` for lexical matching on exact numbers, SLA terms, etc.
- Qdrant point IDs are ``uint64``; Python string IDs are converted via a
  deterministic hash (``_elem_to_id``).
- A lightweight in-memory cache (``_cache``) stores the full
  :class:`~core.models.AtomicElement` objects so that ``search`` can return
  the *same* instance that was upserted rather than reconstructing from the
  Qdrant payload.
- The Qdrant payload mirrors every field needed to reconstruct an
  :class:`~core.models.AtomicElement` when the cache is cold.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from config.settings import settings
from core.exceptions import VectorStoreError
from core.interfaces import IVectorStore
from core.models import AtomicElement, ElementType
from .embedder import BGEEmbedder

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight BM25-style sparse vector generator
# ---------------------------------------------------------------------------

# Simple tokenizer: lowercase, split on non-alphanumeric, keep tokens >= 2 chars
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*", re.IGNORECASE)

# Common English stopwords to exclude from BM25 (keeps the sparse vectors lean)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "be", "as", "by", "this", "that", "with", "from",
    "are", "was", "were", "been", "has", "have", "had", "do", "does",
    "did", "will", "shall", "should", "would", "could", "may", "can",
    "not", "no", "if", "so", "than", "its", "also", "into",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens, removing stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _build_sparse_vector(text: str) -> SparseVector:
    """
    Build a BM25-inspired sparse vector from text.

    Uses term frequency with log-normalization as weights.
    Token indices are deterministic hashes of the token string mapped to
    a 32-bit integer space, which is how Qdrant's sparse vectors work.
    """
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector(indices=[0], values=[0.1])

    tf = Counter(tokens)
    indices: list[int] = []
    values: list[float] = []

    for token, count in tf.items():
        # Deterministic index from token hash (positive int32)
        idx = abs(int(hashlib.md5(token.encode()).hexdigest(), 16)) % (2**31)
        # Log-normalized term frequency
        weight = 1.0 + math.log(1 + count)
        indices.append(idx)
        values.append(round(weight, 4))

    return SparseVector(indices=indices, values=values)


class QdrantVectorStore(IVectorStore):
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        self._embedder: BGEEmbedder = embedder or BGEEmbedder()
        api_key = settings.qdrant_api_key or None
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=api_key,
        )
        self._collection: str = collection_name or settings.qdrant_collection
        self._cache: dict[str, AtomicElement] = {}
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with hybrid vector config if it does not exist.

        If the collection exists but uses the old single-vector schema, it is
        dropped and recreated with the new named-vector schema.  This is safe
        because the graph is rebuilt from scratch on each pipeline run.
        """
        try:
            existing_names = [
                c.name for c in self._client.get_collections().collections
            ]

            if self._collection in existing_names:
                # Check if the existing collection has the hybrid schema
                try:
                    info = self._client.get_collection(self._collection)
                    vectors_config = info.config.params.vectors
                    # If it's using old-style single vector (not a dict), recreate
                    if not isinstance(vectors_config, dict):
                        logger.info(
                            "Recreating Qdrant collection '%s' with hybrid vector schema",
                            self._collection,
                        )
                        self._client.delete_collection(self._collection)
                    else:
                        return  # Already has the right schema
                except Exception:
                    return  # Can't determine schema; leave as-is

            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(),
                },
            )
            logger.info(
                "Created Qdrant collection '%s' with dense + BM25 sparse vectors",
                self._collection,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to ensure Qdrant collection '{self._collection}': {exc}"
            ) from exc

    @staticmethod
    def _elem_to_id(element_id: str) -> int:
        """Return a stable uint63 point ID for *element_id*.

        Python's built-in ``hash()`` is randomised per-process (PYTHONHASHSEED),
        so it changes on every server restart. Using MD5 gives a deterministic,
        collision-resistant mapping that survives restarts and ensures upserts
        overwrite the correct existing point rather than creating a duplicate.
        """
        return int(hashlib.md5(element_id.encode("utf-8")).hexdigest(), 16) % (2 ** 63)

    @staticmethod
    def _payload_to_element(payload: dict) -> AtomicElement:
        raw_type = payload.get("type", ElementType.REQUIREMENT.value)
        try:
            etype = ElementType(raw_type)
        except ValueError:
            etype = ElementType.REQUIREMENT
        elem = AtomicElement(
            id=payload.get("element_id", ""),
            type=etype,
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            document_id=payload.get("document_id", ""),
            confidence=float(payload.get("confidence", 1.0)),
        )
        elem.metadata["section"] = payload.get("section", "")
        elem.metadata["page_number"] = payload.get("page_number", 0)
        return elem

    def upsert(self, elements: list[AtomicElement]) -> None:
        """Encode *elements* and index them in Qdrant with both dense and sparse vectors.

        Parameters
        ----------
        elements:
            Elements to encode and store.  An empty list is a no-op.

        Raises
        ------
        VectorStoreError
            If encoding or the Qdrant upsert call fails.
        """
        if not elements:
            return
        try:
            vectors = self._embedder.embed([e.text for e in elements])
            points = []
            for elem, vec in zip(elements, vectors):
                elem.embedding = vec
                self._cache[elem.id] = elem

                # Build sparse BM25 vector
                sparse = _build_sparse_vector(elem.text)

                points.append(
                    PointStruct(
                        id=self._elem_to_id(elem.id),
                        vector={
                            "dense": vec,
                            "bm25": sparse,
                        },
                        payload={
                            "element_id": elem.id,
                            "type": elem.type.value,
                            "text": elem.text,
                            "source": elem.source,
                            "document_id": elem.document_id,
                            "confidence": elem.confidence,
                            "section": elem.metadata.get("section", ""),
                            "page_number": elem.metadata.get("page_number", 0),
                            "content_hash": elem.metadata.get("content_hash", ""),
                        },
                    )
                )

            self._client.upsert(collection_name=self._collection, points=points)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Upsert failed: {exc}") from exc

    def search(self, query: str, n_results: int = 5) -> list[AtomicElement]:
        """Return the *n_results* most similar elements using hybrid dense+sparse search.

        Uses Qdrant's RRF (Reciprocal Rank Fusion) to combine dense semantic
        similarity with sparse BM25 lexical matching.

        Parameters
        ----------
        query:
            Natural-language query string.
        n_results:
            Maximum number of results to return.

        Returns
        -------
        list[AtomicElement]
            Ordered by hybrid relevance score.

        Raises
        ------
        VectorStoreError
            If the embed or search call fails.
        """
        try:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            qvec: list[float] = self._embedder.embed_one(query)
            q_sparse = _build_sparse_vector(query)

            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(
                        query=qvec,
                        using="dense",
                        limit=n_results * 2,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=q_sparse.indices,
                            values=q_sparse.values,
                        ),
                        using="bm25",
                        limit=n_results * 2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=n_results,
                with_payload=True,
            )

            results: list[AtomicElement] = []
            for hit in response.points:
                elem_id: str = hit.payload.get("element_id", "")
                if elem_id in self._cache:
                    results.append(self._cache[elem_id])
                else:
                    results.append(self._payload_to_element(hit.payload))
            return results

        except ImportError:
            # Fallback: if Prefetch/Fusion not available in this qdrant-client version,
            # fall back to dense-only search
            logger.warning("Hybrid search not available, falling back to dense-only")
            return self._search_dense_only(query, n_results)
        except Exception as exc:
            # Fallback on any hybrid search error
            logger.warning("Hybrid search failed (%s), falling back to dense-only", exc)
            return self._search_dense_only(query, n_results)

    def _search_dense_only(self, query: str, n_results: int = 5) -> list[AtomicElement]:
        """Fallback: dense-only search when hybrid is unavailable."""
        try:
            qvec: list[float] = self._embedder.embed_one(query)
            response = self._client.query_points(
                collection_name=self._collection,
                query=qvec,
                using="dense",
                limit=n_results,
                with_payload=True,
            )
            results = []
            for hit in response.points:
                elem_id = hit.payload.get("element_id", "")
                results.append(self._cache.get(elem_id) or self._payload_to_element(hit.payload))
            return results
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Dense search failed: {exc}") from exc

    def search_by_type(
        self, query: str, element_type: ElementType,
        n_results: int = 5, section: Optional[str] = None,
    ) -> list[AtomicElement]:
        """Like :meth:`search` but restricted to a single :class:`ElementType`.

        Parameters
        ----------
        section:
            Optional section label to further filter results.
        """
        try:
            qvec: list[float] = self._embedder.embed_one(query)
            must = [FieldCondition(key="type", match=MatchValue(value=element_type.value))]
            if section is not None:
                must.append(FieldCondition(key="section", match=MatchValue(value=section)))
            
            response = self._client.query_points(
                collection_name=self._collection,
                query=qvec,
                using="dense",
                limit=n_results,
                with_payload=True,
                query_filter=Filter(must=must),
            )
            results = []
            for hit in response.points:
                elem_id = hit.payload.get("element_id", "")
                results.append(self._cache.get(elem_id) or self._payload_to_element(hit.payload))
            return results
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Search by type failed: {exc}") from exc

    def check_element_exists(self, content_hash: str) -> bool:
        """
        Check if an element with the given content hash already exists in Qdrant.

        Used for granular chunk-level deduplication.

        Parameters
        ----------
        content_hash:
            SHA-256 hash (truncated) of the element text content.

        Returns
        -------
        bool
            True if a matching element exists.
        """
        try:
            response = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="content_hash",
                            match=MatchValue(value=content_hash),
                        )
                    ]
                ),
                limit=1,
            )
            points, _ = response
            return len(points) > 0
        except Exception:
            return False

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection)
            self._cache.clear()
            self._ensure_collection()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Clear failed: {exc}") from exc
