"""
Neo4j-backed graph store — workspace-scoped.

Every method that reads or writes Element nodes takes a ``workspace_id``
parameter so that multiple workspaces can coexist in the same database
without cross-contamination.

Constraint migration
--------------------
On first start the setup routine drops the old single-property ``id``
uniqueness constraint (if present) and creates a composite
``(id, workspace_id)`` unique constraint instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase, Driver

from core.models import AtomicElement, Relationship, ElementType, RelationshipType
from core.interfaces import IGraphStore
from core.exceptions import GraphStoreError
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jGraphStore(IGraphStore):
    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._db: str = settings.neo4j_database
        self._setup_constraints()

    def _setup_constraints(self) -> None:
        with self._driver.session(database=self._db) as s:
            # Drop legacy single-property id constraint if it exists
            try:
                result = s.run(
                    "SHOW CONSTRAINTS WHERE labelsOrTypes = ['Element'] "
                    "AND properties = ['id'] AND type = 'UNIQUENESS'"
                )
                for row in result:
                    s.run(f"DROP CONSTRAINT `{row['name']}` IF EXISTS")
            except Exception:
                pass

            # Composite uniqueness: same element id can exist in different workspaces
            s.run(
                "CREATE CONSTRAINT element_workspace_unique IF NOT EXISTS "
                "FOR (e:Element) REQUIRE (e.id, e.workspace_id) IS UNIQUE"
            )
            s.run("CREATE INDEX element_workspace IF NOT EXISTS FOR (e:Element) ON (e.workspace_id)")
            s.run("CREATE INDEX element_type IF NOT EXISTS FOR (e:Element) ON (e.type)")
            s.run("CREATE INDEX element_document IF NOT EXISTS FOR (e:Element) ON (e.document_id)")
            s.run("CREATE INDEX element_section IF NOT EXISTS FOR (e:Element) ON (e.section)")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_element(self, element: AtomicElement, workspace_id: str) -> None:
        query = (
            "MERGE (e:Element {id: $id, workspace_id: $wid}) "
            "SET e.type = $type, "
            "    e.text = $text, "
            "    e.source = $source, "
            "    e.document_id = $document_id, "
            "    e.confidence = $confidence, "
            "    e.metadata = $metadata, "
            "    e.section = $section"
        )
        try:
            with self._driver.session(database=self._db) as s:
                s.run(query,
                    id=element.id, wid=workspace_id,
                    type=element.type.value, text=element.text,
                    source=element.source, document_id=element.document_id,
                    confidence=element.confidence,
                    metadata=str(element.metadata),
                    section=element.metadata.get("section", ""),
                )
        except Exception as exc:
            raise GraphStoreError(f"Failed to add element '{element.id}': {exc}") from exc

    def add_relationship(self, rel: Relationship, workspace_id: str) -> None:
        rel_type_str = rel.type.value
        query = (
            f"MATCH (a:Element {{id: $src, workspace_id: $wid}}), "
            f"      (b:Element {{id: $tgt, workspace_id: $wid}}) "
            f"MERGE (a)-[r:{rel_type_str}]->(b) "
            f"SET r.confidence = $conf, r.evidence = $ev"
        )
        try:
            with self._driver.session(database=self._db) as s:
                s.run(query, src=rel.source_id, tgt=rel.target_id,
                      wid=workspace_id, conf=rel.confidence, ev=rel.evidence)
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to add relationship {rel.source_id!r} -[{rel_type_str}]-> "
                f"{rel.target_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Batch write operations  (significantly faster than one-by-one)
    # ------------------------------------------------------------------

    def add_elements_batch(
        self, elements: list[AtomicElement], workspace_id: str
    ) -> None:
        """Insert or update all *elements* in a single UNWIND transaction.

        Opens one session, sends one query with a list parameter, and lets
        Neo4j handle the looping server-side.  For 500 elements this reduces
        ~500 round-trips to exactly one.

        Parameters
        ----------
        elements:
            Elements to insert or update (MERGE semantics).
        workspace_id:
            Workspace scope applied to every node.
        """
        if not elements:
            return
        batch = [
            {
                "id": e.id,
                "type": e.type.value,
                "text": e.text,
                "source": e.source,
                "document_id": e.document_id,
                "confidence": e.confidence,
                "metadata": str(e.metadata),
                "section": e.metadata.get("section", ""),
            }
            for e in elements
        ]
        query = (
            "UNWIND $batch AS props "
            "MERGE (e:Element {id: props.id, workspace_id: $wid}) "
            "SET e.type       = props.type, "
            "    e.text       = props.text, "
            "    e.source     = props.source, "
            "    e.document_id = props.document_id, "
            "    e.confidence = props.confidence, "
            "    e.metadata   = props.metadata, "
            "    e.section    = props.section"
        )
        try:
            with self._driver.session(database=self._db) as s:
                s.run(query, batch=batch, wid=workspace_id)
        except Exception as exc:
            raise GraphStoreError(
                f"Batch element insert failed for workspace '{workspace_id}': {exc}"
            ) from exc

    def add_relationships_batch(
        self, relationships: list[Relationship], workspace_id: str
    ) -> None:
        """Insert or update all *relationships* grouped by type, one query per type.

        Neo4j requires the relationship type to be a static string in the Cypher
        query — it cannot be a runtime parameter.  We therefore group rels by
        type and emit one batched UNWIND query per distinct type, which is still
        dramatically faster than one session-per-relationship.

        Parameters
        ----------
        relationships:
            Relationships to insert or update (MERGE semantics).
        workspace_id:
            Workspace scope for the MATCH of both endpoint nodes.
        """
        if not relationships:
            return

        # Group by relationship type to allow static type names in Cypher
        from collections import defaultdict
        by_type: dict[str, list[dict]] = defaultdict(list)
        for rel in relationships:
            by_type[rel.type.value].append({
                "src": rel.source_id,
                "tgt": rel.target_id,
                "conf": rel.confidence,
                "ev": rel.evidence,
            })

        try:
            with self._driver.session(database=self._db) as s:
                for rel_type_str, batch in by_type.items():
                    query = (
                        f"UNWIND $batch AS rel "
                        f"MATCH (a:Element {{id: rel.src, workspace_id: $wid}}), "
                        f"      (b:Element {{id: rel.tgt, workspace_id: $wid}}) "
                        f"MERGE (a)-[r:{rel_type_str}]->(b) "
                        f"SET r.confidence = rel.conf, r.evidence = rel.ev"
                    )
                    s.run(query, batch=batch, wid=workspace_id)
        except Exception as exc:
            raise GraphStoreError(
                f"Batch relationship insert failed for workspace '{workspace_id}': {exc}"
            ) from exc


    def clear_workspace(self, workspace_id: str) -> None:
        """Delete all elements (and their relationships) belonging to workspace_id."""
        try:
            with self._driver.session(database=self._db) as s:
                s.run("MATCH (e:Element {workspace_id: $wid}) DETACH DELETE e", wid=workspace_id)
        except Exception as exc:
            raise GraphStoreError(f"Failed to clear workspace '{workspace_id}': {exc}") from exc

    def clear_document(self, document_id: str, workspace_id: str) -> None:
        """Remove all elements belonging to document_id within workspace_id."""
        try:
            with self._driver.session(database=self._db) as s:
                s.run(
                    "MATCH (e:Element {document_id: $did, workspace_id: $wid}) DETACH DELETE e",
                    did=document_id, wid=workspace_id,
                )
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to clear document '{document_id}' in workspace '{workspace_id}': {exc}"
            ) from exc

    def clear_non_contains_relationships(self, workspace_id: str) -> None:
        """Delete all semantic relationships (everything except CONTAINS) for a workspace.
        Called before re-writing a fresh coordinator batch so stale rels don't accumulate."""
        try:
            with self._driver.session(database=self._db) as s:
                s.run(
                    "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) "
                    "WHERE type(r) <> 'CONTAINS' DELETE r",
                    wid=workspace_id,
                )
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to clear relationships for workspace '{workspace_id}': {exc}"
            ) from exc

    def clear(self) -> None:
        """Delete ALL nodes across ALL workspaces — admin use only."""
        try:
            with self._driver.session(database=self._db) as s:
                s.run("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            raise GraphStoreError(f"Failed to clear graph: {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_element(self, element_id: str, workspace_id: str) -> Optional[AtomicElement]:
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (e:Element {id: $id, workspace_id: $wid}) RETURN e",
                    id=element_id, wid=workspace_id,
                )
                record = result.single()
                return self._node_to_element(record["e"]) if record else None
        except Exception as exc:
            raise GraphStoreError(f"Failed to fetch element '{element_id}': {exc}") from exc

    def get_element_doc_id(self, element_id: str, workspace_id: str) -> str:
        """Return the document ID that CONTAINS this element (via explicit edge).
        Falls back to the element's stored document_id property if no CONTAINS edge exists."""
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (doc:Element {type: 'Document', workspace_id: $wid})"
                    "-[:CONTAINS]->(e:Element {id: $id, workspace_id: $wid}) "
                    "RETURN doc.id AS doc_id",
                    id=element_id, wid=workspace_id,
                )
                record = result.single()
                if record and record["doc_id"]:
                    return record["doc_id"]
                # Fallback: read the property directly
                result2 = s.run(
                    "MATCH (e:Element {id: $id, workspace_id: $wid}) RETURN e.document_id AS doc_id",
                    id=element_id, wid=workspace_id,
                )
                rec2 = result2.single()
                return (rec2["doc_id"] or "") if rec2 else ""
        except Exception:
            return ""

    def get_all_elements(self, workspace_id: str) -> list[AtomicElement]:
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (e:Element {workspace_id: $wid}) WHERE e.type <> 'Document' RETURN e",
                    wid=workspace_id,
                )
                return [self._node_to_element(r["e"]) for r in result]
        except Exception as exc:
            raise GraphStoreError(f"Failed to fetch all elements: {exc}") from exc

    def get_elements_by_type(self, element_type: ElementType, workspace_id: str) -> list[AtomicElement]:
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (e:Element {type: $type, workspace_id: $wid}) RETURN e",
                    type=element_type.value, wid=workspace_id,
                )
                return [self._node_to_element(r["e"]) for r in result]
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to fetch elements of type '{element_type.value}': {exc}"
            ) from exc

    def get_relationships(self, element_id: str, workspace_id: str) -> list[Relationship]:
        try:
            seen: set[tuple[str, str, str]] = set()
            rels: list[Relationship] = []
            with self._driver.session(database=self._db) as s:
                for r in s.run(
                    "MATCH (a:Element {id: $id, workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) "
                    "RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    "       r.confidence AS conf, r.evidence AS ev",
                    id=element_id, wid=workspace_id,
                ):
                    key = (r["src"], r["rtype"], r["tgt"])
                    if key not in seen:
                        seen.add(key)
                        rels.append(self._record_to_relationship(r))
                for r in s.run(
                    "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {id: $id, workspace_id: $wid}) "
                    "RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    "       r.confidence AS conf, r.evidence AS ev",
                    id=element_id, wid=workspace_id,
                ):
                    key = (r["src"], r["rtype"], r["tgt"])
                    if key not in seen:
                        seen.add(key)
                        rels.append(self._record_to_relationship(r))
            return rels
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to fetch relationships for '{element_id}': {exc}"
            ) from exc

    def get_incoming_relationships(
        self, element_id: str, workspace_id: str,
        rel_type: Optional[RelationshipType] = None,
    ) -> list[Relationship]:
        try:
            if rel_type is not None:
                query = (
                    f"MATCH (a:Element {{workspace_id: $wid}})-[r:{rel_type.value}]->"
                    f"(b:Element {{id: $id, workspace_id: $wid}}) "
                    f"RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    f"       r.confidence AS conf, r.evidence AS ev"
                )
            else:
                query = (
                    "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {id: $id, workspace_id: $wid}) "
                    "RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    "       r.confidence AS conf, r.evidence AS ev"
                )
            with self._driver.session(database=self._db) as s:
                return [self._record_to_relationship(r)
                        for r in s.run(query, id=element_id, wid=workspace_id)]
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to fetch incoming relationships for '{element_id}': {exc}"
            ) from exc

    def get_outgoing_relationships(
        self, element_id: str, workspace_id: str,
        rel_type: Optional[RelationshipType] = None,
    ) -> list[Relationship]:
        try:
            if rel_type is not None:
                query = (
                    f"MATCH (a:Element {{id: $id, workspace_id: $wid}})-[r:{rel_type.value}]->(b:Element {{workspace_id: $wid}}) "
                    f"RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    f"       r.confidence AS conf, r.evidence AS ev"
                )
            else:
                query = (
                    "MATCH (a:Element {id: $id, workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) "
                    "RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
                    "       r.confidence AS conf, r.evidence AS ev"
                )
            with self._driver.session(database=self._db) as s:
                return [self._record_to_relationship(r)
                        for r in s.run(query, id=element_id, wid=workspace_id)]
        except Exception as exc:
            raise GraphStoreError(
                f"Failed to fetch outgoing relationships for '{element_id}': {exc}"
            ) from exc

    def get_type_counts(self, workspace_id: str) -> dict[str, int]:
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (e:Element {workspace_id: $wid}) RETURN e.type AS t, count(e) AS cnt",
                    wid=workspace_id,
                )
                return {r["t"]: r["cnt"] for r in result}
        except Exception:
            return {}

    def get_cross_document_relationships(self, workspace_id: str) -> list[dict]:
        """Return all non-CONTAINS edges that cross document boundaries."""
        query = (
            "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) "
            "WHERE type(r) <> 'CONTAINS' "
            "  AND a.document_id IS NOT NULL AND b.document_id IS NOT NULL "
            "  AND a.document_id <> '' AND b.document_id <> '' "
            "  AND a.document_id <> b.document_id "
            "RETURN "
            "  a.id AS src_id, a.type AS src_type, a.text AS src_text, "
            "  a.source AS src_source, a.document_id AS src_doc, "
            "  type(r) AS rtype, coalesce(r.confidence, 1.0) AS conf, "
            "  coalesce(r.evidence, '') AS ev, "
            "  b.id AS tgt_id, b.type AS tgt_type, b.text AS tgt_text, "
            "  b.source AS tgt_source, b.document_id AS tgt_doc"
        )
        try:
            with self._driver.session(database=self._db) as s:
                return [dict(r) for r in s.run(query, wid=workspace_id)]
        except Exception as exc:
            raise GraphStoreError(f"Failed to fetch cross-document relationships: {exc}") from exc

    def get_document_hashes(self, workspace_id: str) -> dict[str, str]:
        try:
            with self._driver.session(database=self._db) as s:
                result = s.run(
                    "MATCH (e:Element {type: 'Document', workspace_id: $wid}) "
                    "WHERE e.doc_hash IS NOT NULL AND e.doc_hash <> '' "
                    "RETURN e.id AS doc_id, e.doc_hash AS doc_hash",
                    wid=workspace_id,
                )
                return {r["doc_id"]: r["doc_hash"] for r in result}
        except Exception as exc:
            raise GraphStoreError(f"Failed to fetch document hashes: {exc}") from exc

    def get_graph_for_visualization(self, workspace_id: str) -> dict:
        nodes_query = (
            "MATCH (e:Element {workspace_id: $wid}) "
            "RETURN e.id AS id, e.type AS type, e.text AS text, "
            "       e.source AS source, e.document_id AS doc_id"
        )
        edges_query = (
            "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) "
            "RETURN a.id AS src, type(r) AS rtype, b.id AS tgt, "
            "       r.confidence AS conf, coalesce(r.evidence, '') AS ev"
        )
        try:
            with self._driver.session(database=self._db) as s:
                nodes = [dict(r) for r in s.run(nodes_query, wid=workspace_id)]
                edges = [dict(r) for r in s.run(edges_query, wid=workspace_id)]
            return {"nodes": nodes, "edges": edges}
        except Exception as exc:
            raise GraphStoreError(f"Failed to fetch graph for visualization: {exc}") from exc

    def node_count(self, workspace_id: str) -> int:
        with self._driver.session(database=self._db) as s:
            result = s.run(
                "MATCH (e:Element {workspace_id: $wid}) RETURN count(e) AS cnt",
                wid=workspace_id,
            )
            return result.single()["cnt"]

    def edge_count(self, workspace_id: str) -> int:
        with self._driver.session(database=self._db) as s:
            result = s.run(
                "MATCH (a:Element {workspace_id: $wid})-[r]->(b:Element {workspace_id: $wid}) RETURN count(r) AS cnt",
                wid=workspace_id,
            )
            return result.single()["cnt"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _node_to_element(self, node: object) -> AtomicElement:
        props = dict(node)  # type: ignore[call-overload]
        raw_type = props.get("type", "")
        try:
            elem_type = ElementType(raw_type)
        except ValueError:
            elem_type = ElementType.REQUIREMENT

        raw_meta = props.get("metadata", "{}")
        try:
            import ast
            metadata: dict = ast.literal_eval(raw_meta) if raw_meta else {}
        except Exception:
            metadata = {}

        if props.get("section"):
            metadata["section"] = props["section"]
        if props.get("page_number") is not None:
            metadata["page_number"] = props["page_number"]

        return AtomicElement(
            id=props["id"],
            type=elem_type,
            text=props.get("text", ""),
            source=props.get("source", ""),
            document_id=props.get("document_id") or "",
            confidence=float(props.get("confidence", 1.0)),
            metadata=metadata,
        )

    @staticmethod
    def _record_to_relationship(record: object) -> Relationship:
        r = dict(record)  # type: ignore[call-overload]
        try:
            rel_type = RelationshipType(r.get("rtype", ""))
        except ValueError:
            rel_type = RelationshipType.CONTAINS
        return Relationship(
            source_id=r["src"],
            target_id=r["tgt"],
            type=rel_type,
            confidence=float(r.get("conf") or 1.0),
            evidence=r.get("ev") or "",
        )
