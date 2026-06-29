"""
LLM-powered extraction of typed atomic elements and relationships.

Implements :class:`core.interfaces.IExtractor` using OpenAI GPT-4o
function calling for structured output.

Changes from the original:
- Token-based chunking via ``RecursiveCharacterTextSplitter.from_tiktoken_encoder``
  replaces the character-based sentence splitter for consistent LLM context sizes.
- Section-aware grouping and context prefix ``[Section | Page N]`` are preserved.
- Deterministic UUID v5 IDs replace sequential counters for idempotent re-ingestion.
- GPT-4o Vision analysis for extracted images (charts, tables, diagrams).
"""

import base64
import hashlib
import json
import logging
import re
import uuid
from typing import Callable

from openai import OpenAI

from config.settings import settings
from core.models import (
    AtomicElement,
    ImageContext,
    ParsedDocument,
    Relationship,
    ElementType,
    RelationshipType,
)
from core.interfaces import IExtractor
from core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-based splitter (lazy import to avoid import-time overhead)
# ---------------------------------------------------------------------------

_SPLITTER = None


def _get_token_splitter():
    """
    Lazy-load the tiktoken-based recursive text splitter.

    Uses ``cl100k_base`` encoding (the encoding used by GPT-4o) with
    1024-token chunks and 200-token overlap.
    """
    global _SPLITTER
    if _SPLITTER is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1024,
            chunk_overlap=200,
        )
    return _SPLITTER


# ---------------------------------------------------------------------------
# OpenAI tool schemas (unchanged)
# ---------------------------------------------------------------------------
ELEMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_elements",
        "description": "Extract typed atomic procurement elements",
        "parameters": {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["Requirement", "Clause", "Risk", "Mitigation", "LD"],
                            },
                            "text": {"type": "string"},
                            "source": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["id", "type", "text", "source", "confidence"],
                    },
                }
            },
            "required": ["elements"],
        },
    },
}

RELATIONSHIP_TOOL = {
    "type": "function",
    "function": {
        "name": "create_relationships",
        "description": "Infer relationships between elements",
        "parameters": {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "COVERS",
                                    "PARTIALLY_COVERS",
                                    "INTRODUCES_RISK",
                                    "MITIGATED_BY",
                                    "LINKED_TO_LD",
                                    "CONTRADICTS",
                                ],
                            },
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["source_id", "target_id", "type", "confidence", "evidence"],
                    },
                }
            },
            "required": ["relationships"],
        },
    },
}

# ---------------------------------------------------------------------------
# UUID v5 namespace for deterministic element IDs
# ---------------------------------------------------------------------------

_GRAPHRAG_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class LLMExtractor(IExtractor):
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SECTION_RE = re.compile(
        r'(?:^|\n)(?:'
        r'Section\s+\d+[\.:]?\s+\w'           # Section 1. Letter
        r'|(?:\d+\.){1,3}\s*\w'                # 3.1.2 Background
        r'|(?:APPENDIX|ANNEX)\s+[A-Z"\']+'     # APPENDIX A / ANNEX "B"
        r'|(?:GCC|SCC)\s+\d+\.\d+'             # GCC 6.1
        r'|[IVX]+\.\s+[A-Z][A-Z]'              # IV. APPENDICES
        r')',
        re.IGNORECASE | re.MULTILINE,
    )

    def _detect_section(self, page_text: str) -> str | None:
        """Scan the first 5 lines of a page for a section header.

        Returns the full matched line (not just the regex match) so labels
        like 'Section 5. Terms of Reference' are preserved intact.
        """
        lines = page_text.splitlines()[:5]
        for line in lines:
            if self._SECTION_RE.search(line):
                label = line.strip()
                # Truncate very long labels (e.g. inline paragraph text mistaken for header)
                return label[:80] if len(label) > 80 else label
        return None

    def _chunk_pages(
        self, pages: list[str]
    ) -> list[tuple[str, str, int]]:
        """
        Chunk a list of pages into (section_label, chunk_text, start_page_1indexed) tuples.

        Section headers are detected from the first 5 lines of each page.
        The last known section label is carried forward to pages with no header.

        Now uses token-based splitting via ``RecursiveCharacterTextSplitter``
        with tiktoken's ``cl100k_base`` encoding for consistent LLM context.
        Every chunk is prefixed with ``[{section_label} | Page {start_page}]``
        so the LLM has structural context.
        """
        # Group pages by section
        current_section = "General"
        # Each group: (section_label, start_page_1indexed, accumulated_text)
        groups: list[tuple[str, int, str]] = []

        for page_idx, page_text in enumerate(pages):
            page_num = page_idx + 1  # 1-indexed
            detected = self._detect_section(page_text)
            if detected:
                current_section = detected
            # Start a new group when section changes or on first page
            if not groups or groups[-1][0] != current_section:
                groups.append((current_section, page_num, page_text))
            else:
                # Append to existing group's text
                label, start, accumulated = groups[-1]
                groups[-1] = (label, start, accumulated + "\n\n" + page_text)

        # Now split each group into token-based chunks
        splitter = _get_token_splitter()
        result: list[tuple[str, str, int]] = []

        for section_label, start_page, section_text in groups:
            raw_chunks = splitter.split_text(section_text)
            for raw_chunk in raw_chunks:
                # Filter out chunks with < 80 chars of actual content
                if len(raw_chunk.strip()) < 80:
                    continue
                prefix = f"[{section_label} | Page {start_page}]\n"
                chunk_text = prefix + raw_chunk
                result.append((section_label, chunk_text, start_page))

        return result

    @staticmethod
    def _generate_element_id(
        doc_hash: str,
        section: str,
        chunk_idx: int,
        elem_type: str,
        elem_idx: int,
    ) -> str:
        """
        Generate a deterministic, human-readable element ID using UUID v5.

        The UUID is derived from a composite key of the document hash,
        section label, chunk index, element type, and element index.
        The final ID uses a short prefix + truncated UUID for readability:
        e.g. ``REQ_a3b4c5d6`` instead of ``REQ_001``.
        """
        raw = f"{doc_hash}_{section}_{chunk_idx}_{elem_type}_{elem_idx}"
        uid = uuid.uuid5(_GRAPHRAG_NS, raw)
        short = uid.hex[:8]  # 8 hex chars = 32 bits, 4B+ unique IDs
        prefix_map = {
            "Requirement": "REQ",
            "Clause": "CL",
            "Risk": "RISK",
            "Mitigation": "MIT",
            "LD": "LD",
        }
        pfx = prefix_map.get(elem_type, "ELM")
        return f"{pfx}_{short}"

    def _type_str_to_enum(self, t: str) -> ElementType:
        mapping: dict[str, ElementType] = {
            "Requirement": ElementType.REQUIREMENT,
            "Clause": ElementType.CLAUSE,
            "Risk": ElementType.RISK,
            "Mitigation": ElementType.MITIGATION,
            "LD": ElementType.LD,
        }
        return mapping.get(t, ElementType.REQUIREMENT)

    def _rel_str_to_enum(self, t: str) -> RelationshipType | None:
        try:
            return RelationshipType(t)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # IExtractor interface
    # ------------------------------------------------------------------

    def extract_elements(
        self,
        doc: ParsedDocument,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[AtomicElement]:
        # Compute a content hash for deterministic IDs
        content_hash = hashlib.sha256(
            "\n".join(doc.pages).encode("utf-8")
        ).hexdigest()[:12]

        chunks = self._chunk_pages(doc.pages)
        raw_elements: list[dict] = []
        # chunk_meta maps a list index to (section_label, start_page, chunk_idx)
        chunk_meta: list[tuple[str, int, int]] = []
        counters: dict[str, int] = {t: 0 for t in ["REQ", "CL", "RISK", "MIT", "LD"]}
        prefix_map: dict[str, str] = {
            "Requirement": "REQ",
            "Clause": "CL",
            "Risk": "RISK",
            "Mitigation": "MIT",
            "LD": "LD",
        }

        # Doc-slug prefix for human readability in Neo4j
        raw_slug = re.sub(r"^DOC_", "", doc.id).upper()
        doc_slug = re.sub(r"[^A-Z0-9]", "", raw_slug)[:4] or "DOC"

        if progress_cb:
            progress_cb(f"  {len(chunks)} section-chunk(s) to process via {settings.llm_model}")

        for chunk_idx, (section_label, chunk_text, start_page) in enumerate(chunks):
            if progress_cb:
                progress_cb(
                    f"  LLM [{chunk_idx + 1}/{len(chunks)}] {section_label[:50]} (p.{start_page})"
                    f" — {len(chunk_text)} chars"
                )

            source_hint = f"{doc.name} — {section_label}"
            system = (
                f"You are a procurement document analyst. Extract atomic semantic elements.\n"
                f"Document type: {doc.type.value} | Document: {doc.name}\n\n"
                f"Element types:\n"
                f"- Requirement: any measurable obligation the contractor/consultant MUST fulfil — "
                f"deliverables with deadlines, hard copies/electronic copies, reporting frequency, "
                f"personnel qualifications, SLA targets, compliance mandates. Common in RFP/TOR.\n"
                f"- Clause: contractual term from a contract/agreement — payment schedules, "
                f"advance payments, termination rights, arbitration, warranty periods, liability caps.\n"
                f"- Risk: explicit potential negative outcome or breach scenario.\n"
                f"- Mitigation: action or mechanism that reduces a specific risk.\n"
                f"- LD: Liquidated Damages or financial penalty tied to non-performance.\n\n"
                f"Extraction rules:\n"
                f"- Extract EVERY distinct obligation, deadline, or deliverable as its own element.\n"
                f"- For tables: each row with distinct data should be extracted as a separate element.\n"
                f"- For scanned/OCR'd text: interpret imperfectly formatted lines (OCR artefacts) — "
                f"focus on the semantic meaning, not the exact formatting.\n"
                f"- Ignore pure table headers, form template labels, and blank-fill instructions.\n"
                f"- ID format: use simple sequential IDs like REQ_001, CL_001 etc.\n"
                f"- confidence: 0.9+ if explicitly stated, 0.7–0.9 if implied, skip below 0.7\n"
                f'- Source: use "{source_hint}" verbatim as the source field'
            )

            try:
                resp = self.client.chat.completions.create(
                    model=settings.llm_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Extract elements from:\n\n{chunk_text}"},
                    ],
                    tools=[ELEMENT_TOOL],
                    tool_choice={"type": "function", "function": {"name": "extract_elements"}},
                    max_tokens=settings.max_tokens_extraction,
                )
                tc = resp.choices[0].message.tool_calls
                n_found = 0
                if tc:
                    data = json.loads(tc[0].function.arguments)
                    for elem_idx, e in enumerate(data.get("elements", [])):
                        if e.get("confidence", 0) >= settings.confidence_threshold:
                            raw_elements.append(e)
                            chunk_meta.append((section_label, start_page, chunk_idx))
                            pfx = prefix_map.get(e["type"], "REQ")
                            counters[pfx] += 1
                            n_found += 1
                if progress_cb:
                    usage = resp.usage
                    tok_info = (
                        f" ({usage.prompt_tokens}+{usage.completion_tokens} tok)"
                        if usage else ""
                    )
                    progress_cb(
                        f"  ✓ [{chunk_idx + 1}/{len(chunks)}] {n_found} element(s) above threshold{tok_info}"
                    )
            except Exception as ex:
                if progress_cb:
                    progress_cb(f"  ✗ [{chunk_idx + 1}/{len(chunks)}] extraction failed: {ex}")
                raise ExtractionError(
                    f"Element extraction failed on chunk {chunk_idx}: {ex}"
                ) from ex

        # Deduplicate: within same type, if word overlap > 70% keep higher confidence
        deduped: list[dict] = []
        deduped_meta: list[tuple[str, int, int]] = []
        for idx, elem in enumerate(raw_elements):
            words_e = set(elem["text"].lower().split())
            duplicate = False
            for j, kept in enumerate(deduped):
                if kept["type"] == elem["type"]:
                    words_k = set(kept["text"].lower().split())
                    union = words_e | words_k
                    if union and len(words_e & words_k) / len(union) > 0.7:
                        if elem.get("confidence", 0) > kept.get("confidence", 0):
                            deduped[j] = elem
                            deduped_meta[j] = chunk_meta[idx]
                        else:
                            duplicate = True
                        break
            if not duplicate:
                deduped.append(elem)
                deduped_meta.append(chunk_meta[idx])

        if progress_cb and len(raw_elements) != len(deduped):
            progress_cb(
                f"  Dedup: {len(raw_elements)} raw → {len(deduped)} unique elements"
            )

        # Build AtomicElement objects with deterministic UUID v5 IDs
        # The IDs are prefixed with a doc slug for human readability:
        # e.g. UTAH_REQ_a3b4c5d6
        type_counters: dict[str, int] = {}
        result: list[AtomicElement] = []
        for e, (section_label, page_number, chunk_idx) in zip(deduped, deduped_meta):
            elem_type = e["type"]
            type_counters[elem_type] = type_counters.get(elem_type, 0) + 1
            elem_idx = type_counters[elem_type]

            deterministic_id = self._generate_element_id(
                content_hash, section_label, chunk_idx, elem_type, elem_idx
            )
            elem_id = f"{doc_slug}_{deterministic_id}"

            atomic = AtomicElement(
                id=elem_id,
                type=self._type_str_to_enum(elem_type),
                text=e["text"],
                source=e.get("source", doc.name),
                document_id=doc.id,
                confidence=float(e.get("confidence", 1.0)),
            )
            atomic.metadata["section"] = section_label
            atomic.metadata["page_number"] = page_number
            atomic.metadata["content_hash"] = hashlib.sha256(
                e["text"].encode("utf-8")
            ).hexdigest()[:16]
            result.append(atomic)
        return result

    def extract_relationships(self, elements: list[AtomicElement]) -> list[Relationship]:
        if not elements:
            return []

        # Sort by ID so the LLM always sees the same input order regardless of
        # how Neo4j returns elements (traversal order varies between workspaces).
        ordered = sorted(elements, key=lambda e: e.id)
        elem_ids = {e.id for e in ordered}

        # Group by document so the LLM sees all elements from each document together —
        # this dramatically improves COVERS/PARTIALLY_COVERS detection between documents.
        by_doc: dict[str, list[AtomicElement]] = {}
        for e in ordered:
            by_doc.setdefault(e.document_id, []).append(e)

        sections: list[str] = []
        for doc_id, doc_elems in sorted(by_doc.items()):
            header = f"=== Document: {doc_elems[0].source.split(' — ')[0] if doc_elems else doc_id} ==="
            lines = [header] + [
                f"{e.id} | {e.type.value} | {e.text[:120]} ({e.source})"
                for e in doc_elems
            ]
            sections.append("\n".join(lines))
        elem_list = "\n\n".join(sections)

        system = (
            "You are a procurement knowledge graph analyst. "
            "Infer ALL applicable relationships between the elements below.\n\n"
            "Relationship types with MANDATORY source→target direction "
            "(never reverse these):\n\n"
            "1. COVERS           source=Clause        → target=Requirement\n"
            "   The Contract Clause fully satisfies the Requirement (same/better SLA or terms).\n\n"
            "2. PARTIALLY_COVERS source=Clause        → target=Requirement\n"
            "   The Clause addresses the same topic but with weaker SLA or missing aspects.\n\n"
            "3. INTRODUCES_RISK  source=Requirement   → target=Risk\n"
            "   If this Requirement is not met or breached, it creates this Risk.\n\n"
            "4. MITIGATED_BY     source=Risk          → target=Mitigation\n"
            "   This Mitigation action reduces or controls the Risk.\n\n"
            "5. LINKED_TO_LD     source=Risk or Requirement → target=LD\n"
            "   This LD (Liquidated Damages clause) is the financial consequence.\n\n"
            "6. CONTRADICTS      source=Clause        → target=Clause\n"
            "   The two Clauses directly conflict with each other.\n\n"
            "Scan ALL pairs across documents AND within the same document. "
            "Only add a relationship if the semantic connection is genuinely present in the "
            "element texts — do NOT invent relationships to fill gaps.\n\n"
            "Rules: confidence >= 0.6 only. Both IDs must exist in the provided list."
        )

        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Infer relationships for:\n\n{elem_list}"},
                ],
                tools=[RELATIONSHIP_TOOL],
                tool_choice={"type": "function", "function": {"name": "create_relationships"}},
                max_tokens=8000,
            )
            tc = resp.choices[0].message.tool_calls
            if not tc:
                return []

            # Build a quick type lookup so we can enforce direction constraints below.
            elem_type_map = {e.id: e.type for e in ordered}

            data = json.loads(tc[0].function.arguments)
            rels: list[Relationship] = []
            for r in data.get("relationships", []):
                src_id, tgt_id = r["source_id"], r["target_id"]
                if src_id not in elem_ids or tgt_id not in elem_ids:
                    continue
                if float(r.get("confidence", 0)) < settings.confidence_threshold:
                    continue
                rel_type = self._rel_str_to_enum(r["type"])
                if rel_type is None:
                    continue

                # Enforce direction for all relationship types.
                # If the LLM reverses any of these, auto-flip rather than silently losing data.
                src_t = elem_type_map.get(src_id)
                tgt_t = elem_type_map.get(tgt_id)
                should_flip = False

                if rel_type in (RelationshipType.COVERS, RelationshipType.PARTIALLY_COVERS):
                    # Clause → Requirement
                    if src_t == ElementType.REQUIREMENT and tgt_t == ElementType.CLAUSE:
                        should_flip = True
                elif rel_type == RelationshipType.INTRODUCES_RISK:
                    # Requirement → Risk
                    if src_t == ElementType.RISK and tgt_t == ElementType.REQUIREMENT:
                        should_flip = True
                elif rel_type == RelationshipType.MITIGATED_BY:
                    # Risk → Mitigation
                    if src_t == ElementType.MITIGATION and tgt_t == ElementType.RISK:
                        should_flip = True
                elif rel_type == RelationshipType.LINKED_TO_LD:
                    # Risk/Requirement → LD
                    if src_t == ElementType.LD and tgt_t in (
                        ElementType.RISK, ElementType.REQUIREMENT
                    ):
                        should_flip = True

                if should_flip:
                    src_id, tgt_id = tgt_id, src_id
                    logger.debug(
                        "Auto-flipped %s %s→%s to correct direction",
                        rel_type.value, r["source_id"], r["target_id"],
                    )

                rels.append(
                    Relationship(
                        source_id=src_id,
                        target_id=tgt_id,
                        type=rel_type,
                        confidence=float(r["confidence"]),
                        evidence=r.get("evidence", ""),
                    )
                )
            return rels
        except Exception as ex:
            raise ExtractionError(f"Relationship extraction failed: {ex}") from ex

    # ------------------------------------------------------------------
    # GPT-4o Vision — image analysis
    # ------------------------------------------------------------------

    def analyze_images(
        self, images: list[ImageContext], doc: ParsedDocument
    ) -> list[AtomicElement]:
        """
        Send extracted images to GPT-4o Vision for structured analysis.

        Each image's OCR text and spatial context are sent along with the
        base64-encoded image.  GPT-4o returns a structured summary which
        is then converted into AtomicElement objects and fed into the
        standard extraction pipeline.

        Parameters
        ----------
        images:
            List of :class:`ImageContext` objects from PDF image extraction.
        doc:
            The parent document (used for document_id and source).

        Returns
        -------
        list[AtomicElement]
            Elements extracted from the image analysis.
        """
        if not images:
            return []

        elements: list[AtomicElement] = []
        raw_slug = re.sub(r"^DOC_", "", doc.id).upper()
        doc_slug = re.sub(r"[^A-Z0-9]", "", raw_slug)[:4] or "DOC"

        for img_idx, img_ctx in enumerate(images):
            try:
                # Encode image to base64
                b64_image = base64.b64encode(img_ctx.image_bytes).decode("utf-8")

                context_parts = []
                if img_ctx.surrounding_text:
                    context_parts.append(
                        f"Surrounding document text:\n{img_ctx.surrounding_text}"
                    )
                if img_ctx.ocr_text:
                    context_parts.append(
                        f"OCR-extracted text from the image:\n{img_ctx.ocr_text}"
                    )

                context_str = "\n\n".join(context_parts) if context_parts else ""

                system_prompt = (
                    "You are a procurement document analyst examining an image from a "
                    f"{doc.type.value} document named '{doc.name}'.\n\n"
                    "Analyze the image and extract atomic procurement elements.\n"
                    "For tables: extract each row with distinct obligations, SLAs, or requirements.\n"
                    "For charts/diagrams: describe the key metrics and thresholds.\n\n"
                    "Return your analysis using the extract_elements function.\n"
                    "Element types: Requirement, Clause, Risk, Mitigation, LD.\n"
                    "Use sequential IDs: REQ_001, CL_001, etc.\n"
                    f'Source: "Image on Page {img_ctx.page_number} of {doc.name}"'
                )

                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ]
                if context_str:
                    user_content.insert(0, {"type": "text", "text": context_str})

                resp = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    tools=[ELEMENT_TOOL],
                    tool_choice={"type": "function", "function": {"name": "extract_elements"}},
                    max_tokens=settings.max_tokens_extraction,
                )

                tc = resp.choices[0].message.tool_calls
                if not tc:
                    continue

                data = json.loads(tc[0].function.arguments)
                for elem_idx, e in enumerate(data.get("elements", [])):
                    if e.get("confidence", 0) < settings.confidence_threshold:
                        continue

                    # Deterministic ID from image hash + index
                    det_id = self._generate_element_id(
                        img_ctx.image_hash, "image", img_idx, e["type"], elem_idx
                    )
                    elem_id = f"{doc_slug}_{det_id}"

                    atomic = AtomicElement(
                        id=elem_id,
                        type=self._type_str_to_enum(e["type"]),
                        text=e["text"],
                        source=e.get("source", f"Image on Page {img_ctx.page_number} of {doc.name}"),
                        document_id=doc.id,
                        confidence=float(e.get("confidence", 1.0)),
                    )
                    atomic.metadata["section"] = f"Image on Page {img_ctx.page_number}"
                    atomic.metadata["page_number"] = img_ctx.page_number
                    atomic.metadata["from_image"] = True
                    atomic.metadata["image_hash"] = img_ctx.image_hash
                    elements.append(atomic)

            except Exception as exc:
                logger.warning(
                    "Vision analysis failed for image %d on page %d: %s",
                    img_idx, img_ctx.page_number, exc,
                )
                continue

        logger.info(
            "GPT-4o Vision extracted %d elements from %d images",
            len(elements), len(images),
        )
        return elements
