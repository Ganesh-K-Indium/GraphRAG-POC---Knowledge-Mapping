"""
High-level graph construction and analysis — workspace-scoped.

All methods take ``workspace_id`` so that multiple workspaces can share
the same Neo4jGraphStore singleton without cross-contamination.
"""
from __future__ import annotations

import logging
from typing import Any

from core.models import (
    AtomicElement, CoverageResult, CoverageStatus,
    ElementType, ParsedDocument, Relationship, RelationshipType,
)
from .neo4j_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, store: Neo4jGraphStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_elements(
        self,
        elements: list[AtomicElement],
        relationships: list[Relationship],
        documents: list[ParsedDocument],
        workspace_id: str,
        doc_hashes: dict[str, str] | None = None,
    ) -> None:
        logger.info(
            "Building graph [%s]: %d docs, %d elements, %d relationships",
            workspace_id, len(documents), len(elements), len(relationships),
        )

        # Evict only the documents being (re-)ingested within this workspace
        for doc in documents:
            self.store.clear_document(doc.id, workspace_id)

        # ── Document pseudo-nodes (one per doc, written individually; small count) ──
        _hashes = doc_hashes or {}
        for doc in documents:
            with self.store._driver.session(database=self.store._db) as s:
                s.run(
                    "MERGE (e:Element {id: $id, workspace_id: $wid}) "
                    "SET e.type = 'Document', "
                    "    e.text = $text, "
                    "    e.source = $src, "
                    "    e.document_id = $did, "
                    "    e.confidence = 1.0, "
                    "    e.doc_hash = $hash",
                    id=doc.id, wid=workspace_id,
                    text=doc.name, src=doc.type.value,
                    did=doc.id, hash=_hashes.get(doc.id, ""),
                )

        # ── Batch insert all extracted elements in one round-trip ──
        self.store.add_elements_batch(elements, workspace_id)

        # ── Batch insert CONTAINS edges (Document → Element) in one round-trip ──
        contains_rels = [
            Relationship(
                source_id=elem.document_id,
                target_id=elem.id,
                type=RelationshipType.CONTAINS,
                confidence=1.0,
                evidence="Document contains this element",
            )
            for elem in elements
        ]
        self.store.add_relationships_batch(contains_rels, workspace_id)

        # Clear all semantic (non-CONTAINS) relationships before writing the fresh coordinator
        # batch. This prevents stale rels from previous incremental runs from accumulating —
        # e.g. a COVERS rel found in run 1 that the LLM no longer finds in run 2 would otherwise
        # silently persist and inflate coverage metrics.
        self.store.clear_non_contains_relationships(workspace_id)

        # ── Batch insert semantic relationships in one round-trip per type ──
        self.store.add_relationships_batch(relationships, workspace_id)

        logger.info(
            "Graph build complete [%s]. Nodes: %d, Edges: %d",
            workspace_id,
            self.store.node_count(workspace_id),
            self.store.edge_count(workspace_id),
        )


    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    def assess_coverage(self, workspace_id: str) -> list[CoverageResult]:
        requirements = self.store.get_elements_by_type(ElementType.REQUIREMENT, workspace_id)
        results: list[CoverageResult] = []

        for req in requirements:
            full_coverage = [
                r.source_id for r in self.store.get_incoming_relationships(
                    req.id, workspace_id, RelationshipType.COVERS)
            ]
            partial_coverage = [
                r.source_id for r in self.store.get_incoming_relationships(
                    req.id, workspace_id, RelationshipType.PARTIALLY_COVERS)
            ]
            risks = [
                r.target_id for r in self.store.get_outgoing_relationships(
                    req.id, workspace_id, RelationshipType.INTRODUCES_RISK)
            ]
            mitigations: list[str] = []
            lds: list[str] = [
                r.target_id for r in self.store.get_outgoing_relationships(
                    req.id, workspace_id, RelationshipType.LINKED_TO_LD)
            ]
            for risk_id in risks:
                mitigations.extend(
                    r.target_id for r in self.store.get_outgoing_relationships(
                        risk_id, workspace_id, RelationshipType.MITIGATED_BY)
                )
                lds.extend(
                    r.target_id for r in self.store.get_outgoing_relationships(
                        risk_id, workspace_id, RelationshipType.LINKED_TO_LD)
                )

            if full_coverage:
                status = CoverageStatus.COVERED
            elif partial_coverage:
                status = CoverageStatus.PARTIAL
            else:
                status = CoverageStatus.NOT_COVERED

            results.append(CoverageResult(
                requirement_id=req.id,
                requirement_text=req.text,
                status=status,
                covering_clauses=full_coverage + partial_coverage,
                risks=risks,
                mitigations=list(set(mitigations)),
                lds=list(set(lds)),
                source=req.source,
            ))

        return results

    # ------------------------------------------------------------------
    # Traceability
    # ------------------------------------------------------------------

    def _enrich(self, element_id: str, relationship: str,
                req_document_id: str, workspace_id: str) -> dict[str, Any] | None:
        elem = self.store.get_element(element_id, workspace_id)
        if elem is None:
            return None
        # Use the CONTAINS edge as the authoritative source for document ownership.
        # The stored document_id property is a fallback for elements that pre-date the edge.
        elem_doc_id = self.store.get_element_doc_id(element_id, workspace_id) or elem.document_id
        is_inter = (
            bool(elem_doc_id) and bool(req_document_id)
            and elem_doc_id != req_document_id
        )
        return {
            "id": elem.id,
            "type": elem.type.value if hasattr(elem.type, "value") else str(elem.type),
            "text": elem.text,
            "source": elem.source,
            "document_id": elem_doc_id,
            "relationship": relationship,
            "is_inter_document": is_inter,
        }

    def get_traceability_chain(self, req_id: str, workspace_id: str) -> dict[str, Any]:
        req = self.store.get_element(req_id, workspace_id)
        if req is None:
            return {}

        req_doc_id = self.store.get_element_doc_id(req_id, workspace_id) or req.document_id

        full_ids = [r.source_id for r in self.store.get_incoming_relationships(
            req_id, workspace_id, RelationshipType.COVERS)]
        partial_ids = [r.source_id for r in self.store.get_incoming_relationships(
            req_id, workspace_id, RelationshipType.PARTIALLY_COVERS)]
        risk_ids = [r.target_id for r in self.store.get_outgoing_relationships(
            req_id, workspace_id, RelationshipType.INTRODUCES_RISK)]

        mitigation_ids: list[str] = []
        ld_ids = [r.target_id for r in self.store.get_outgoing_relationships(
            req_id, workspace_id, RelationshipType.LINKED_TO_LD)]

        for risk_id in risk_ids:
            mitigation_ids.extend(r.target_id for r in self.store.get_outgoing_relationships(
                risk_id, workspace_id, RelationshipType.MITIGATED_BY))
            ld_ids.extend(r.target_id for r in self.store.get_outgoing_relationships(
                risk_id, workspace_id, RelationshipType.LINKED_TO_LD))

        gaps: list[str] = []
        if not full_ids and not partial_ids:
            gaps.append("No contract clause covers this requirement")
        if risk_ids and not mitigation_ids:
            gaps.append(f"Risks {risk_ids} have no mitigation")
        if risk_ids and not ld_ids:
            gaps.append(f"Risks {risk_ids} have no Liquidated Damages")

        def enrich_list(ids: list[str], rel: str) -> list[dict[str, Any]]:
            return [e for eid in ids
                    if (e := self._enrich(eid, rel, req_doc_id, workspace_id)) is not None]

        def enrich_dedup(ids: list[str], rel: str) -> list[dict[str, Any]]:
            seen: set[str] = set()
            out = []
            for eid in ids:
                if eid in seen:
                    continue
                seen.add(eid)
                e = self._enrich(eid, rel, req_doc_id, workspace_id)
                if e:
                    out.append(e)
            return out

        return {
            "requirement": {
                "id": req.id,
                "type": req.type.value if hasattr(req.type, "value") else str(req.type),
                "text": req.text, "source": req.source,
                "document_id": req_doc_id,
                "relationship": "", "is_inter_document": False,
            },
            "full_coverage": enrich_list(full_ids, "COVERS"),
            "partial_coverage": enrich_list(partial_ids, "PARTIALLY_COVERS"),
            "risks": enrich_list(risk_ids, "INTRODUCES_RISK"),
            "mitigations": enrich_dedup(mitigation_ids, "MITIGATED_BY"),
            "lds": enrich_dedup(ld_ids, "LINKED_TO_LD"),
            "gaps": gaps,
        }
