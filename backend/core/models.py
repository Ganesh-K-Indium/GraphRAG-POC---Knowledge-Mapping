"""
Domain models for the GraphRAG POC knowledge mapping system.

All models are plain Python dataclasses (no Pydantic dependency).
Covers the full element / relationship vocabulary for RFP, Risk Sheet,
and Contract document types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    """Supported source document categories."""

    RFP = "RFP"
    RISK_SHEET = "RiskSheet"
    CONTRACT = "Contract"


class ElementType(str, Enum):
    """Atomic knowledge-graph node types extracted from documents."""

    REQUIREMENT = "Requirement"
    CLAUSE = "Clause"
    RISK = "Risk"
    MITIGATION = "Mitigation"
    LD = "LD"  # Liquidated Damages


class RelationshipType(str, Enum):
    """Typed directed edges in the knowledge graph."""

    CONTAINS = "CONTAINS"
    COVERS = "COVERS"
    PARTIALLY_COVERS = "PARTIALLY_COVERS"
    INTRODUCES_RISK = "INTRODUCES_RISK"
    MITIGATED_BY = "MITIGATED_BY"
    LINKED_TO_LD = "LINKED_TO_LD"
    CONTRADICTS = "CONTRADICTS"


class CoverageStatus(str, Enum):
    """High-level coverage verdict for a single requirement."""

    COVERED = "Covered"
    PARTIAL = "Partially Covered"
    NOT_COVERED = "Not Covered"


# ---------------------------------------------------------------------------
# Image / media context (extracted from PDF pages)
# ---------------------------------------------------------------------------


@dataclass
class ImageContext:
    """
    Represents an image extracted from a PDF page along with its OCR text
    and spatial-proximity context.

    This is a *parallel* data structure — it does not change the
    :class:`ParsedDocument` interface.  Image contexts are passed alongside
    parsed documents through the extraction pipeline.

    Attributes
    ----------
    image_bytes:
        Raw PNG bytes of the preprocessed (enhanced) image.
    page_number:
        1-indexed page number where the image was found.
    ocr_text:
        Text extracted from the image via Tesseract OCR.
    surrounding_text:
        Text blocks in spatial proximity to the image on the same page.
    image_hash:
        SHA-256 hex digest of the raw image bytes for deduplication.
    source_filename:
        Name of the PDF this image was extracted from.
    """

    image_bytes: bytes
    page_number: int
    ocr_text: str = ""
    surrounding_text: str = ""
    image_hash: str = ""
    source_filename: str = ""


# ---------------------------------------------------------------------------
# Core graph elements
# ---------------------------------------------------------------------------


@dataclass
class AtomicElement:
    """
    A single extracted node in the knowledge graph.

    ``id`` must be globally unique within a processing session and stable
    across reloads (use a deterministic hash or UUID stored alongside the
    document, not a random UUID generated at runtime).

    ``embedding`` is intentionally excluded from ``repr`` to keep log output
    readable; it is populated by the vector-store layer.
    """

    id: str
    type: ElementType
    text: str
    source: str          # Human-readable location, e.g. "RFP Page 4"
    document_id: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r} "
                f"for element '{self.id}'"
            )


@dataclass
class Relationship:
    """
    A typed directed edge between two :class:`AtomicElement` nodes.

    ``evidence`` is a short natural-language justification produced by the
    LLM extractor explaining why this relationship was inferred.
    """

    source_id: str
    target_id: str
    type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r} "
                f"for relationship {self.source_id!r} -> {self.target_id!r}"
            )


# ---------------------------------------------------------------------------
# Document representation
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """
    Raw parsed representation of a source document.

    ``pages`` holds the extracted text for each page (1-indexed by
    convention: ``pages[0]`` is page 1).  For documents without page
    boundaries a single-element list is acceptable.
    """

    id: str
    name: str
    type: DocumentType
    pages: List[str]        # One entry per page; may be empty strings for blanks
    total_pages: int

    def __post_init__(self) -> None:
        if self.total_pages != len(self.pages):
            raise ValueError(
                f"total_pages ({self.total_pages}) does not match "
                f"len(pages) ({len(self.pages)}) for document '{self.id}'"
            )


# ---------------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """
    Bundled output from a single document extraction pass.

    Relationships here are *intra*-document; cross-document relationships
    are resolved at the graph-merge stage.
    """

    document_id: str
    elements: List[AtomicElement] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coverage analysis output
# ---------------------------------------------------------------------------


@dataclass
class CoverageResult:
    """
    Coverage verdict for a single :class:`~ElementType.REQUIREMENT`.

    ``covering_clauses``, ``risks``, ``mitigations``, and ``lds`` store
    element IDs (not full objects) to keep the dataclass lightweight.
    Callers resolve them against the graph store as needed.
    """

    requirement_id: str
    requirement_text: str
    status: CoverageStatus
    covering_clauses: List[str] = field(default_factory=list)   # Clause element IDs
    risks: List[str] = field(default_factory=list)              # Risk element IDs
    mitigations: List[str] = field(default_factory=list)        # Mitigation element IDs
    lds: List[str] = field(default_factory=list)                # LD element IDs
    source: str = ""                                            # Human-readable origin


# ---------------------------------------------------------------------------
# Knowledge graph container
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeGraph:
    """
    In-memory representation of the full knowledge graph.

    ``elements`` is keyed by :attr:`AtomicElement.id`.
    ``documents`` is keyed by :attr:`ParsedDocument.id`.

    This container is intentionally thin — persistence, querying, and
    vector-search are delegated to the store implementations defined in
    :mod:`core.interfaces`.
    """

    elements: Dict[str, AtomicElement] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    documents: Dict[str, ParsedDocument] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_element(self, element: AtomicElement) -> None:
        """Insert or replace an element by ID."""
        self.elements[element.id] = element

    def add_relationship(self, rel: Relationship) -> None:
        """Append a relationship, skipping exact duplicates."""
        if rel not in self.relationships:
            self.relationships.append(rel)

    def add_document(self, doc: ParsedDocument) -> None:
        """Insert or replace a parsed document by ID."""
        self.documents[doc.id] = doc

    def get_elements_by_type(self, element_type: ElementType) -> List[AtomicElement]:
        """Return all elements of a given type."""
        return [e for e in self.elements.values() if e.type == element_type]

    def get_relationships_for(self, element_id: str) -> List[Relationship]:
        """Return all relationships where *element_id* is source or target."""
        return [
            r for r in self.relationships
            if r.source_id == element_id or r.target_id == element_id
        ]

    def stats(self) -> Dict[str, Any]:
        """Return a summary dict suitable for logging or display."""
        type_counts: Dict[str, int] = {}
        for e in self.elements.values():
            type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1

        rel_counts: Dict[str, int] = {}
        for r in self.relationships:
            rel_counts[r.type.value] = rel_counts.get(r.type.value, 0) + 1

        return {
            "documents": len(self.documents),
            "elements": len(self.elements),
            "elements_by_type": type_counts,
            "relationships": len(self.relationships),
            "relationships_by_type": rel_counts,
        }
