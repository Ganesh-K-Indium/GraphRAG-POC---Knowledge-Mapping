"""
Graph service — workspace-scoped facade over Neo4j, Qdrant, and GraphBuilder.

A GraphService instance is bound to one workspace_id. The underlying
Neo4jGraphStore is a shared singleton (one driver pool); only Qdrant gets a
per-workspace collection. GraphitiMemory is omitted per-workspace for now.
"""
from __future__ import annotations

import logging
from typing import Any

from core.models import AtomicElement, CoverageResult, ParsedDocument, Relationship
from graph import GraphBuilder, Neo4jGraphStore
from vector import QdrantVectorStore

# GraphitiMemory is disabled until workspace-isolation is implemented.
# from graph.graphiti_memory import GraphitiMemory

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(
        self,
        workspace_id: str,
        store: Neo4jGraphStore,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.workspace_id = workspace_id
        self.store = store
        self.vector_store = vector_store
        # Graphiti is disabled — uncomment below to re-enable:
        # self.graphiti = GraphitiMemory()
        self.graphiti = None
        self.builder = GraphBuilder(store)
        # GraphVisualizer removed — React frontend handles visualisation.

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_knowledge_graph(
        self,
        documents: list[ParsedDocument],
        elements: list[AtomicElement],
        relationships: list[Relationship],
        doc_hashes: dict[str, str] | None = None,
    ) -> None:
        self.builder.build_from_elements(
            elements, relationships, documents, self.workspace_id, doc_hashes
        )
        self.vector_store.upsert(elements)

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def get_ingested_doc_hashes(self) -> dict[str, str]:
        try:
            return self.store.get_document_hashes(self.workspace_id)
        except Exception:
            return {}

    def reset_workspace(self) -> None:
        self.store.clear_workspace(self.workspace_id)
        self.vector_store.clear()
        logger.warning("Workspace '%s' reset: all Neo4j nodes and Qdrant vectors deleted",
                       self.workspace_id)

    # ------------------------------------------------------------------
    # Coverage & traceability
    # ------------------------------------------------------------------

    def get_coverage_results(self) -> list[CoverageResult]:
        return self.builder.assess_coverage(self.workspace_id)

    def get_traceability(self, req_id: str) -> dict[str, Any]:
        return self.builder.get_traceability_chain(req_id, self.workspace_id)

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def get_all_elements(self) -> list[AtomicElement]:
        return self.store.get_all_elements(self.workspace_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_node_count(self) -> int:
        return self.store.node_count(self.workspace_id)

    def get_edge_count(self) -> int:
        return self.store.edge_count(self.workspace_id)

    def close(self) -> None:
        self.store.close()
