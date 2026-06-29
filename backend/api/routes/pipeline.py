"""
Pipeline route — workspace-scoped, SSE streaming.

POST /api/workspaces/{workspace_id}/pipeline/run

Coordination:
    Each workspace gets its own _Coordinator instance.  Concurrent pipeline
    runs within the same workspace share a combined cross-document
    relationship extraction batch.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import queue as _queue
import time
import uuid
from typing import AsyncGenerator

import logging

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from api.deps import get_doc_service, get_graph_service

router = APIRouter(prefix="/api/workspaces/{workspace_id}")

_COORD_WINDOW: float = 6.0


@dataclasses.dataclass
class _ExtractionResult:
    run_id: str
    elements: list
    docs: list
    hashes: dict
    done: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    relationships: list = dataclasses.field(default_factory=list)
    batch_size: int = 0


class _Coordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending: list[_ExtractionResult] = []
        self._timer: asyncio.Task | None = None

    async def submit_and_wait(self, result: _ExtractionResult, workspace_id: str) -> None:
        async with self._lock:
            self._pending.append(result)
            if self._timer and not self._timer.done():
                self._timer.cancel()
            self._timer = asyncio.create_task(
                self._fire_after_quiescence(workspace_id)
            )
        await result.done.wait()

    async def pending_count(self) -> int:
        async with self._lock:
            return len(self._pending)

    async def _fire_after_quiescence(self, workspace_id: str) -> None:
        await asyncio.sleep(_COORD_WINDOW)
        await self._execute(workspace_id)

    async def _execute(self, workspace_id: str) -> None:
        async with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()
            self._timer = None

        doc_svc = get_doc_service()
        graph_svc = get_graph_service(workspace_id)

        all_new = [e for r in batch for e in r.elements]
        new_ids = {e.id for e in all_new}

        try:
            existing = await asyncio.to_thread(graph_svc.store.get_all_elements, workspace_id)
            combined = all_new + [e for e in existing if e.id not in new_ids]
            relationships = await asyncio.to_thread(
                doc_svc.extract_cross_document_relationships, combined
            )
        except Exception as exc:
            logger.warning("Coordinator cross-doc extraction failed for workspace %s: %s", workspace_id, exc)
            relationships = []

        for r in batch:
            r.relationships = relationships
            r.batch_size = len(batch)
            r.done.set()


# Per-workspace coordinator instances
_coordinators: dict[str, _Coordinator] = {}
_WRITE_LOCKS: dict[str, asyncio.Lock] = {}


def _get_coordinator(workspace_id: str) -> _Coordinator:
    if workspace_id not in _coordinators:
        _coordinators[workspace_id] = _Coordinator()
    return _coordinators[workspace_id]


def _get_write_lock(workspace_id: str) -> asyncio.Lock:
    if workspace_id not in _WRITE_LOCKS:
        _WRITE_LOCKS[workspace_id] = asyncio.Lock()
    return _WRITE_LOCKS[workspace_id]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_pipeline(
    workspace_id: str,
    file_data: list[tuple[bytes, str]],
) -> AsyncGenerator[str, None]:
    t0 = time.perf_counter()
    run_id = str(uuid.uuid4())
    doc_svc = get_doc_service()
    graph_svc = get_graph_service(workspace_id)
    coordinator = _get_coordinator(workspace_id)
    write_lock = _get_write_lock(workspace_id)

    skipped: list[str] = []
    to_process: list[tuple[bytes, str]] = []

    # ── Step 1: Parse + Dedup ───────────────────────────────────────────
    yield _sse({"type": "step_start", "step": "parse", "label": "Parsing Documents",
                "total": len(file_data)})
    await asyncio.sleep(0)

    try:
        existing_hashes = await asyncio.to_thread(graph_svc.get_ingested_doc_hashes)
        known_hashes: set[str] = set(existing_hashes.values())

        for fb, fn in file_data:
            h = hashlib.sha256(fb).hexdigest()
            size_kb = len(fb) / 1024
            if h in known_hashes:
                skipped.append(fn)
                yield _sse({"type": "step_progress", "step": "parse",
                            "message": f"Skipped '{fn}' ({size_kb:.0f} KB) — already ingested (hash match)",
                            "current": len(skipped), "total": len(file_data)})
            else:
                to_process.append((fb, fn))
                yield _sse({"type": "step_progress", "step": "parse",
                            "message": f"Queued '{fn}' ({size_kb:.0f} KB) for extraction",
                            "current": len(to_process), "total": len(file_data)})
            await asyncio.sleep(0)

        if not to_process and skipped:
            yield _sse({"type": "step_complete", "step": "parse", "count": 0,
                        "elapsed": round(time.perf_counter() - t0, 2)})
            await asyncio.sleep(0)
            for step_id, step_label in [
                ("extract", "Extracting Elements (LLM)"),
                ("graph", "Building Knowledge Graph"),
                ("vector", "Indexing Semantic Vectors"),
            ]:
                yield _sse({"type": "step_start", "step": step_id, "label": step_label, "total": 0})
                await asyncio.sleep(0)
                yield _sse({"type": "step_complete", "step": step_id, "count": 0, "elapsed": 0})
                await asyncio.sleep(0)
            yield _sse({"type": "step_start", "step": "coverage",
                        "label": "Assessing Coverage", "total": 0})
            await asyncio.sleep(0)
            coverage = await asyncio.to_thread(graph_svc.get_coverage_results)
            yield _sse({"type": "step_complete", "step": "coverage",
                        "count": len(coverage), "elapsed": 0})
            await asyncio.sleep(0)
            yield _sse({
                "type": "pipeline_complete",
                "workspace_id": workspace_id,
                "summary": {
                    "documents": 0, "skipped": len(skipped), "elements": 0,
                    "nodes": await asyncio.to_thread(graph_svc.get_node_count),
                    "edges": await asyncio.to_thread(graph_svc.get_edge_count),
                    "coverage_items": len(coverage),
                    "elapsed": round(time.perf_counter() - t0, 2),
                },
            })
            await asyncio.sleep(0)
            return

    except Exception as exc:
        yield _sse({"type": "error", "step": "parse", "message": str(exc)})
        await asyncio.sleep(0)
        return

    # ── Step 2: LLM Extraction — one file at a time for full progress ───
    yield _sse({"type": "step_complete", "step": "parse", "count": len(to_process),
                "elapsed": round(time.perf_counter() - t0, 2)})
    await asyncio.sleep(0)
    yield _sse({"type": "step_start", "step": "extract",
                "label": "Extracting Elements (LLM)", "total": len(to_process)})
    await asyncio.sleep(0)

    t_extract = time.perf_counter()
    docs: list = []
    elements: list = []
    new_hashes: dict = {}

    for qi, (fb, fn) in enumerate(to_process):
        yield _sse({
            "type": "step_progress", "step": "extract",
            "message": f"[{qi + 1}/{len(to_process)}] Processing '{fn}' (text + tables + images + LLM extraction)…",
            "current": qi, "total": len(to_process),
        })
        await asyncio.sleep(0)

        # Stream progress from sync thread in real-time using a queue +
        # background task pattern: drain every 300ms while the thread runs.
        progress_q: _queue.SimpleQueue = _queue.SimpleQueue()

        def _cb(msg: str, q: _queue.SimpleQueue = progress_q) -> None:
            q.put(msg)

        async def _drain(current: int) -> None:
            while not progress_q.empty():
                yield _sse({"type": "step_progress", "step": "extract",
                            "message": progress_q.get_nowait(),
                            "current": current, "total": len(to_process)})

        try:
            t_file = time.perf_counter()
            file_task = asyncio.create_task(
                asyncio.to_thread(doc_svc.process_file, fb, fn, _cb)
            )

            # Drain queue every 300 ms so OCR/LLM progress appears in real-time
            while not file_task.done():
                await asyncio.sleep(0.3)
                async for chunk in _drain(qi):
                    yield chunk

            doc, file_elems = await file_task

        except Exception as exc:
            async for chunk in _drain(qi):
                yield chunk
            yield _sse({"type": "error", "step": "extract", "message": str(exc)})
            await asyncio.sleep(0)
            return

        # Final drain after thread completes
        async for chunk in _drain(qi + 1):
            yield chunk

        file_elapsed = round(time.perf_counter() - t_file, 1)
        file_hash = hashlib.sha256(fb).hexdigest()
        new_hashes[doc.id] = file_hash
        docs.append(doc)
        elements.extend(file_elems)

        type_counts: dict[str, int] = {}
        for e in file_elems:
            type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
        types_str = "  ".join(f"{v}x{k}" for k, v in sorted(type_counts.items()))

        yield _sse({"type": "step_progress", "step": "extract",
                    "message": (
                        f"[{qi + 1}/{len(to_process)}] '{fn}' done — "
                        f"{len(file_elems)} elements · {doc.total_pages}p · {doc.type.value} · {file_elapsed}s"
                        + (f"  [{types_str}]" if types_str else "")
                    ),
                    "current": qi + 1, "total": len(to_process)})
        await asyncio.sleep(0)

    yield _sse({"type": "step_complete", "step": "extract", "count": len(elements),
                "elapsed": round(time.perf_counter() - t_extract, 2)})
    await asyncio.sleep(0)

    if not docs:
        yield _sse({
            "type": "pipeline_complete",
            "workspace_id": workspace_id,
            "summary": {"documents": 0, "elements": 0, "nodes": 0, "edges": 0,
                        "elapsed": round(time.perf_counter() - t0, 2)},
        })
        await asyncio.sleep(0)
        return

    # ── Step 3: Coordinated cross-document relationship extraction ──────
    yield _sse({"type": "step_start", "step": "graph",
                "label": "Building Knowledge Graph", "total": len(elements)})
    await asyncio.sleep(0)
    try:
        t_graph = time.perf_counter()
        result = _ExtractionResult(run_id=run_id, elements=elements,
                                   docs=docs, hashes=new_hashes)

        submit_task = asyncio.create_task(
            coordinator.submit_and_wait(result, workspace_id)
        )

        pending_count = await coordinator.pending_count()
        yield _sse({"type": "step_progress", "step": "graph",
                    "message": f"Queued for cross-document analysis — {pending_count} pipeline(s) in batch, waiting for quiescence window…",
                    "current": 0, "total": len(elements)})
        await asyncio.sleep(0)

        sent_running = False
        while not submit_task.done():
            await asyncio.sleep(2.0)
            if not submit_task.done() and not sent_running:
                sent_running = True
                yield _sse({"type": "step_progress", "step": "graph",
                            "message": "Running cross-document relationship extraction via LLM…",
                            "current": 0, "total": len(elements)})
                await asyncio.sleep(0)

        await submit_task

        relationships = result.relationships
        batch_size = result.batch_size
        batch_note = f" (shared batch of {batch_size} pipelines)" if batch_size > 1 else ""

        rel_type_counts: dict[str, int] = {}
        for r in relationships:
            rt = r.type.value if hasattr(r.type, "value") else str(r.type)
            rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1
        rel_detail = "  ".join(f"{v}x{k}" for k, v in sorted(rel_type_counts.items()))

        yield _sse({"type": "step_progress", "step": "graph",
                    "message": (
                        f"Found {len(relationships)} relationship(s){batch_note}"
                        + (f"  [{rel_detail}]" if rel_detail else "")
                        + " — writing to Neo4j…"
                    ),
                    "current": len(elements), "total": len(elements)})
        await asyncio.sleep(0)

        async with write_lock:
            yield _sse({"type": "step_progress", "step": "graph",
                        "message": f"Merging {len(elements)} nodes + {len(relationships)} edges into Neo4j…",
                        "current": len(elements), "total": len(elements)})
            await asyncio.sleep(0)
            await asyncio.to_thread(
                graph_svc.builder.build_from_elements,
                elements, relationships, docs, workspace_id, new_hashes,
            )
            node_count = await asyncio.to_thread(graph_svc.get_node_count)
            edge_count = await asyncio.to_thread(graph_svc.get_edge_count)

        yield _sse({"type": "step_progress", "step": "graph",
                    "message": f"Graph updated — {node_count} total nodes · {edge_count} total edges in workspace",
                    "current": len(elements), "total": len(elements)})
        await asyncio.sleep(0)

        yield _sse({"type": "step_complete", "step": "graph", "count": node_count,
                    "elapsed": round(time.perf_counter() - t_graph, 2)})
        await asyncio.sleep(0)
    except Exception as exc:
        yield _sse({"type": "error", "step": "graph", "message": str(exc)})
        await asyncio.sleep(0)
        return

    # ── Step 4: Vector Index ────────────────────────────────────────────
    yield _sse({"type": "step_start", "step": "vector",
                "label": "Indexing Semantic Vectors", "total": len(elements)})
    await asyncio.sleep(0)
    try:
        t_vec = time.perf_counter()
        yield _sse({"type": "step_progress", "step": "vector",
                    "message": f"Embedding {len(elements)} elements with BGE-M3 → Qdrant ws_{workspace_id[:8]}…",
                    "current": 0, "total": len(elements)})
        await asyncio.sleep(0)
        async with write_lock:
            await asyncio.to_thread(graph_svc.vector_store.upsert, elements)
        yield _sse({"type": "step_complete", "step": "vector", "count": len(elements),
                    "elapsed": round(time.perf_counter() - t_vec, 2)})
        await asyncio.sleep(0)
    except Exception as exc:
        yield _sse({"type": "error", "step": "vector", "message": str(exc)})
        await asyncio.sleep(0)

    # ── Step 5: Coverage Assessment ─────────────────────────────────────
    yield _sse({"type": "step_start", "step": "coverage",
                "label": "Assessing Coverage", "total": 0})
    await asyncio.sleep(0)
    try:
        t_cov = time.perf_counter()

        all_ws_elements = await asyncio.to_thread(
            graph_svc.store.get_all_elements, workspace_id
        )

        req_count = sum(1 for e in all_ws_elements if e.type.value == "Requirement")
        yield _sse({"type": "step_progress", "step": "coverage",
                    "message": f"Assessing coverage for {req_count} requirement(s)…",
                    "current": 0, "total": 0})
        await asyncio.sleep(0)

        coverage = await asyncio.to_thread(graph_svc.get_coverage_results)

        covered = sum(1 for c in coverage if c.status.value == "Covered")
        partial = sum(1 for c in coverage if c.status.value == "Partially Covered")
        not_cov = sum(1 for c in coverage if c.status.value == "Not Covered")
        yield _sse({"type": "step_progress", "step": "coverage",
                    "message": (
                        f"Coverage result: {covered} covered · {partial} partial · {not_cov} not covered"
                        f" (of {len(coverage)} requirements)"
                    ),
                    "current": 0, "total": 0})
        await asyncio.sleep(0)

        yield _sse({"type": "step_complete", "step": "coverage", "count": len(coverage),
                    "elapsed": round(time.perf_counter() - t_cov, 2)})
        await asyncio.sleep(0)
    except Exception as exc:
        yield _sse({"type": "error", "step": "coverage", "message": str(exc)})
        await asyncio.sleep(0)
        coverage = []

    yield _sse({
        "type": "pipeline_complete",
        "workspace_id": workspace_id,
        "summary": {
            "documents": len(docs), "skipped": len(skipped), "elements": len(elements),
            "nodes": await asyncio.to_thread(graph_svc.get_node_count),
            "edges": await asyncio.to_thread(graph_svc.get_edge_count),
            "coverage_items": len(coverage),
            "elapsed": round(time.perf_counter() - t0, 2),
        },
    })
    await asyncio.sleep(0)


@router.post("/pipeline/run")
async def run_pipeline(
    workspace_id: str,
    files: list[UploadFile] = File(...),
) -> StreamingResponse:
    file_data = [(await f.read(), f.filename or "unknown") for f in files]
    return StreamingResponse(
        _stream_pipeline(workspace_id, file_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
