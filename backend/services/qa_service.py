"""
QA service: intent-aware question-answering over the knowledge graph.

Architecture
------------
:meth:`QAService.answer` follows a **retrieval-augmented generation** pattern:

1. **Intent classification** — a lightweight keyword heuristic routes the
   question to one of four structured-retrieval paths or a general semantic
   fallback.

2. **Evidence gathering** — depending on intent, evidence is pulled from:
   * :class:`~graph.builder.GraphBuilder`  — coverage / traceability queries
   * :class:`~graph.neo4j_store.Neo4jGraphStore`  — graph traversal
   * :class:`~vector.qdrant_store.QdrantVectorStore`  — semantic similarity
   * :class:`~graph.graphiti_memory.GraphitiMemory`  — Graphiti entity search

3. **Synthesis** — the evidence JSON is passed to GPT-4o, which produces a
   concise, cited answer grounded solely in the retrieved evidence.

Intent routes
~~~~~~~~~~~~~
* ``coverage_gap``     — requirements not covered by any contract clause
* ``risk_for_partial`` — risks linked to partially-covered requirements
* ``no_mitigation``    — risks that have no MITIGATED_BY edge
* ``no_ld``            — risks that have no LINKED_TO_LD edge
* ``general``          — semantic vector + Graphiti fallback
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from config.settings import settings
from core.models import (
    AtomicElement,
    CoverageStatus,
    ElementType,
    RelationshipType,
)
from graph.builder import GraphBuilder
# GraphitiMemory is disabled — uncomment to re-enable:
# from graph.graphiti_memory import GraphitiMemory
from graph.neo4j_store import Neo4jGraphStore
from vector.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a procurement knowledge graph analyst. "
    "Answer based only on the provided graph evidence. "
    "Write a concise, direct answer in plain prose — 2 to 4 sentences maximum unless a list is genuinely clearer. "
    "Do NOT include element IDs, source document names, or page references in your answer text. "
    "The user sees the source evidence cards separately, so focus entirely on the substance: "
    "what the data says, what the obligation or risk is, and what the implication is."
)


class QAService:
    """
    Intent-aware question-answering service.

    Parameters
    ----------
    store:
        An open :class:`~graph.neo4j_store.Neo4jGraphStore`.
    builder:
        A :class:`~graph.builder.GraphBuilder` wrapping the same *store*.
    graphiti:
        A :class:`~graph.graphiti_memory.GraphitiMemory` instance (may be
        newly created if not already held by the caller).
    vector_store:
        An open :class:`~vector.qdrant_store.QdrantVectorStore`.

    Notes
    -----
    This service does **not** own the lifetime of the injected dependencies.
    Callers (typically the Streamlit app) are responsible for closing the
    store and vector store when the session ends.
    """

    def __init__(
        self,
        workspace_id: str,
        store: Neo4jGraphStore,
        builder: GraphBuilder,
        graphiti,  # GraphitiMemory | None
        vector_store: QdrantVectorStore,
    ) -> None:
        self.workspace_id = workspace_id
        self.store = store
        self.builder = builder
        self.graphiti = graphiti  # May be None when Graphiti is disabled
        self.vector_store = vector_store
        self._client: OpenAI = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, question: str) -> str:
        """
        Classify a natural-language question into one of five intents.

        The classification is rule-based (keyword matching) to keep latency
        low.  It is deliberately conservative: ambiguous questions fall
        through to the ``"general"`` path which uses vector + Graphiti search.

        Parameters
        ----------
        question:
            The raw question string from the user.

        Returns
        -------
        str
            One of: ``"coverage_gap"``, ``"risk_for_partial"``,
            ``"no_mitigation"``, ``"no_ld"``, ``"general"``.
        """
        q = question.lower()

        if any(
            w in q
            for w in [
                "not covered",
                "uncovered",
                "missing coverage",
                "no coverage",
                "gap",
            ]
        ):
            return "coverage_gap"

        if "risk" in q and any(w in q for w in ["partial", "partially covered"]):
            return "risk_for_partial"

        if any(
            w in q
            for w in [
                "no mitigation",
                "unmitigated",
                "without mitigation",
                "missing mitigation",
            ]
        ):
            return "no_mitigation"

        if any(
            w in q
            for w in [
                "no ld",
                "no liquidated",
                "without ld",
                "missing ld",
                "no penalty",
            ]
        ):
            return "no_ld"

        return "general"

    # ------------------------------------------------------------------
    # Evidence gathering
    # ------------------------------------------------------------------

    def _gather_coverage_gap_evidence(self) -> list[dict[str, Any]]:
        """Return evidence for requirements with no contract-clause coverage."""
        coverage = self.builder.assess_coverage(self.workspace_id)
        uncovered = [c for c in coverage if c.status == CoverageStatus.NOT_COVERED]
        return [
            {
                "id": c.requirement_id,
                "text": c.requirement_text,
                "status": c.status.value,
                "source": c.source,
            }
            for c in uncovered
        ]

    def _gather_risk_for_partial_evidence(self) -> list[dict[str, Any]]:
        """Return risks associated with partially-covered requirements."""
        coverage = self.builder.assess_coverage(self.workspace_id)
        partial = [c for c in coverage if c.status == CoverageStatus.PARTIAL]
        evidence: list[dict[str, Any]] = []
        for c in partial:
            for risk_id in c.risks:
                risk = self.store.get_element(risk_id, self.workspace_id)
                if risk is not None:
                    evidence.append(
                        {
                            "requirement": c.requirement_id,
                            "risk_id": risk_id,
                            "risk_text": risk.text,
                            "source": risk.source,
                        }
                    )
        return evidence

    def _gather_no_mitigation_evidence(self) -> list[dict[str, Any]]:
        """Return risks that have no outgoing MITIGATED_BY relationship."""
        risks = self.store.get_elements_by_type(ElementType.RISK, self.workspace_id)
        evidence: list[dict[str, Any]] = []
        for risk in risks:
            outgoing = self.store.get_outgoing_relationships(risk.id, self.workspace_id)
            has_mitigation = any(
                r.type == RelationshipType.MITIGATED_BY for r in outgoing
            )
            if not has_mitigation:
                evidence.append(
                    {"id": risk.id, "text": risk.text, "source": risk.source}
                )
        return evidence

    def _gather_no_ld_evidence(self) -> list[dict[str, Any]]:
        """Return risks that have no outgoing LINKED_TO_LD relationship."""
        risks = self.store.get_elements_by_type(ElementType.RISK, self.workspace_id)
        evidence: list[dict[str, Any]] = []
        for risk in risks:
            outgoing = self.store.get_outgoing_relationships(risk.id, self.workspace_id)
            has_ld = any(r.type == RelationshipType.LINKED_TO_LD for r in outgoing)
            if not has_ld:
                evidence.append(
                    {"id": risk.id, "text": risk.text, "source": risk.source}
                )
        return evidence

    def _gather_general_evidence(self, question: str) -> list[dict[str, Any]]:
        """
        Gather evidence via semantic vector search and Graphiti entity search.

        The two evidence streams are concatenated; duplicates are acceptable
        since GPT-4o is instructed to de-duplicate in its synthesis.
        """
        evidence: list[dict[str, Any]] = []

        # Vector similarity search (Qdrant)
        try:
            vec_results: list[AtomicElement] = self.vector_store.search(
                question, n_results=5
            )
            for e in vec_results:
                evidence.append(
                    {
                        "id": e.id,
                        "type": e.type.value,
                        "text": e.text,
                        "source": e.source,
                    }
                )
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)

        # Graphiti semantic entity search is disabled (workspace isolation pending).
        # Uncomment the block below to re-enable when Graphiti is configured.
        # try:
        #     if self.graphiti is not None:
        #         graphiti_results = self.graphiti.search_graph_sync(
        #             question, num_results=3
        #         )
        #         for r in graphiti_results:
        #             evidence.append(
        #                 {
        #                     "graphiti_fact": r.get("fact", ""),
        #                     "uuid": r.get("uuid", ""),
        #                 }
        #             )
        # except Exception as exc:
        #     logger.warning("Graphiti search failed: %s", exc)

        return evidence

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    def _synthesize(self, question: str, evidence: list[dict[str, Any]]) -> str:
        """
        Pass the collected evidence to GPT-4o and return the generated answer.

        Parameters
        ----------
        question:
            The original user question.
        evidence:
            Serialisable list of evidence dicts.

        Returns
        -------
        str
            The model-generated answer, or an error message on failure.
        """
        evidence_str = json.dumps(evidence, indent=2, ensure_ascii=False)
        try:
            response = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Graph evidence:\n{evidence_str}"
                        ),
                    },
                ],
                max_tokens=400,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM synthesis failed: %s", exc)
            return f"Could not generate answer: {exc}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict[str, Any]:
        """
        Answer a natural-language question about the knowledge graph.

        Parameters
        ----------
        question:
            Free-text question from the user.

        Returns
        -------
        dict with keys:
            * ``"answer"``     — LLM-generated answer string
            * ``"evidence"``   — list of evidence dicts used for synthesis
            * ``"query_type"`` — the classified intent string
        """
        logger.info("QAService.answer | question=%r", question[:120])

        intent = self._classify_intent(question)
        logger.debug("Classified intent: %s", intent)

        # Gather structured or semantic evidence based on intent
        if intent == "coverage_gap":
            evidence = self._gather_coverage_gap_evidence()
        elif intent == "risk_for_partial":
            evidence = self._gather_risk_for_partial_evidence()
        elif intent == "no_mitigation":
            evidence = self._gather_no_mitigation_evidence()
        elif intent == "no_ld":
            evidence = self._gather_no_ld_evidence()
        else:
            evidence = self._gather_general_evidence(question)

        logger.info(
            "Evidence gathered: %d items (intent=%s)", len(evidence), intent
        )

        answer_text = self._synthesize(question, evidence)

        return {
            "answer": answer_text,
            "evidence": evidence,
            "query_type": intent,
        }
