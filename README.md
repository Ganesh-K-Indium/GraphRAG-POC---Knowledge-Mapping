# KnowledgeMap — Procurement Intelligence Suite

> **A multi-workspace knowledge graph platform for procurement document analysis.**  
> Each workspace is an isolated analysis environment. Upload RFP + Risk Sheet + Contract → automated SSE-streaming pipeline → interactive graph → traceability lineage → natural language Q&A.

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [Project Structure](#project-structure)
7. [UI Walkthrough](#ui-walkthrough)
8. [Extending the System](#extending-the-system)
9. [Troubleshooting](#troubleshooting)

---

## Recent Updates & Optimizations (Nishi)

* **Neo4j Batch Ingestion**: Upgraded graph insertion to use batch `UNWIND` queries in `backend/graph/neo4j_store.py` and `backend/graph/builder.py`, resolving Mac CPU overheating issues during large document pipeline runs (Nishi).
* **Deterministic Vector IDs**: Implemented MD5-hashed UUIDs for vectors in `backend/vector/qdrant_store.py` to prevent duplicate vectors upon container restarts (Nishi).
* **Workspace Isolation Fix**: Fixed cross-workspace data leakage in `backend/services/qa_service.py` by ensuring `workspace_id` propagation (Nishi).
* **DOCX Table Extraction**: Added logic to `backend/parsers/docx_parser.py` to parse XML nodes (`CT_P` and `CT_Tbl`) and convert Word tables into Markdown format, improving extraction quality (Nishi).
* **Centralized SSL Patch**: Consolidated insecure SSL workarounds into a single `core/ssl_patch.py` module loaded at startup to clean up scattered bypass logic (Nishi).
* **Disabled GraphitiMemory**: Temporarily disabled Graphiti inside `QAService` to prevent context bleeding across different workspaces (Nishi).
* **Port Configuration Fix**: Corrected PostgreSQL port mapping in `.env` to align with Docker Compose bindings (Nishi).
* **Conflict Resolution**: Cleanly merged real-time UI progress tracking with the backend optimization branch, resolving conflicts across `llm_extractor.py`, `pipeline.py`, and `document_service.py` (Nishi).

---

## What This Does

The platform lets analysts run multiple independent procurement analyses side-by-side — each in its own workspace with fully isolated graph data, vector indexes, and pipeline state.

| Input | Extracted as |
|-------|-------------|
| RFP / RFX (PDF or DOCX) | `Requirement` nodes |
| Risk Sheet (PDF or DOCX) | `Risk` + `Mitigation` nodes |
| Contract / Offer (PDF or DOCX) | `Clause` + `LD` nodes |

All nodes land in **Neo4j**, scoped to the workspace. Typed edges (`COVERS`, `PARTIALLY_COVERS`, `INTRODUCES_RISK`, `MITIGATED_BY`, `LINKED_TO_LD`, `CONTRADICTS`) connect them across documents. Within each workspace you can:

- Watch a **real-time streaming pipeline** process your documents step by step (SSE)
- Explore an **interactive force-directed / hierarchical knowledge graph** — drag nodes, zoom, expand neighbourhoods
- View **cross-document relationships** in a dedicated sidebar panel with relationship type filtering
- Get a **traceability lineage** — which requirements are covered, partial, or missing, with INTER/INTRA document badges
- Ask **natural language questions** — answered by graph traversal + semantic search, cited to exact sections

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  React Frontend  (Vite · TypeScript · React Router v7)               │
│  localhost:5173                                                        │
│                                                                        │
│  /                     → Workspace Grid (create / open / delete)      │
│  /workspace/:id/:tab   → Workspace App                                │
│                          Ingest · Elements · Graph · Traceability     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP + SSE streaming
┌────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI Backend  (uvicorn · localhost:8000)                          │
│                                                                        │
│  GET/POST /api/workspaces                         Workspace CRUD      │
│  POST /api/workspaces/{id}/pipeline/run           SSE pipeline        │
│  GET  /api/workspaces/{id}/graph/data             React Flow data     │
│  GET  /api/workspaces/{id}/graph/subgraph/:id     1-hop neighbourhood │
│  GET  /api/workspaces/{id}/graph/cross-doc-relationships              │
│  GET  /api/workspaces/{id}/traceability/coverage  Coverage results    │
│  GET  /api/workspaces/{id}/traceability/chain/:id Traceability chain  │
│  POST /api/workspaces/{id}/chat/ask               Intent-aware Q&A   │
│  GET  /api/workspaces/{id}/elements               All elements        │
│  POST /api/workspaces/{id}/reset                  Wipe workspace      │
└───┬──────────────┬──────────────┬────────────────────────────────────┘
    │              │              │
┌───▼───┐      ┌───▼──────┐  ┌───▼──────────────────────────────────┐
│Neo4j  │      │ Qdrant   │  │  PostgreSQL                          │
│:7687  │      │ :6333    │  │  :5432                               │
│       │      │          │  │  workspaces (id, name, desc,         │
│ Per-  │      │ Per-     │  │             created_at, updated_at)  │
│ work- │      │ workspace│  └──────────────────────────────────────┘
│ space │      │ ws_{id}  │
│ nodes │      │          │
└───────┘      └──────────┘
```

### Workspace isolation

| Layer | Isolation mechanism |
|-------|-------------------|
| **PostgreSQL** | One row per workspace — metadata only |
| **Neo4j** | Composite unique constraint `(id, workspace_id) IS UNIQUE`; all queries filter by `workspace_id` |
| **Qdrant** | Separate collection per workspace: `ws_{workspace_id}` |
| **Pipeline** | Per-workspace coordinator and write lock — concurrent runs in the same workspace share one cross-doc extraction batch; different workspaces run fully in parallel. Pipeline state persists across workspace navigation (workspace-scoped Zustand store). |

### Pipeline — five steps (SSE streaming)

```
1  Parse          PDF/DOCX → text pages
                  ┌ Digital PDF  → PyMuPDF native text extraction
                  └ Scanned PDF  → PyMuPDF render at 150 DPI grayscale → Tesseract OCR
                    Per-page OCR progress streams live to the UI
                  Pages with >40% non-ASCII are dropped (garbled OCR filter)
                  SHA-256 dedup skips already-ingested files

2  Extract (LLM)  Section-aware chunking — section headers detected per page
                  Each chunk prefixed [Section | Page N] for structural context
                  GPT-4o function calling (temperature=0) → AtomicElement objects
                  (Requirement / Clause / Risk / Mitigation / LD)
                  IDs are doc-scoped: RFPA_REQ_001, CONT_CL_001

3  Build Graph    Coordinator pattern — concurrent pipeline runs share a single
                  combined cross-doc LLM call (6 s quiescence window)
                  Elements sorted by ID before relationship extraction (deterministic)
                  Relationship types: COVERS / PARTIALLY_COVERS /
                    INTRODUCES_RISK / MITIGATED_BY / LINKED_TO_LD / CONTRADICTS
                  Direction enforced post-LLM (auto-flip guard for all types)
                  Semantic relationships: clear-and-rewrite each run (no stale accumulation)

4  Index Vectors  BGE-M3 embeddings → workspace Qdrant collection
                  Payload: section, page_number — enables section-scoped search

5  Coverage       Graph traversal per Requirement → Covered / Partial / Not Covered
                  All cross-doc relationships already written in Step 3 — no re-extraction
```

### Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 · TypeScript · Vite · React Router v7 · React Flow v12 · d3-force · dagre · Tailwind CSS v3 · Framer Motion · Zustand |
| API | FastAPI · uvicorn · SSE streaming |
| LLM extraction | GPT-4o (OpenAI function calling — structured output) |
| Graph store | Neo4j 5.x (Cypher MERGE, typed edges, composite workspace constraint) |
| Vector store | Qdrant (per-workspace collections) |
| Embeddings | BAAI/bge-m3 (sentence-transformers, 1024-dim) |
| Workspace metadata | PostgreSQL 16 (psycopg2) |
| OCR | Tesseract 5.x + pytesseract + PyMuPDF rendering at 150 DPI grayscale (scanned PDF fallback, per-page live progress) |

---

## Prerequisites

### System requirements

| Tool | Minimum version | Check |
|------|----------------|-------|
| Python | 3.11+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| Docker Desktop | 4.x | `docker --version` |
| Tesseract | 4.x+ | `tesseract --version` |

**Install Tesseract (required for scanned PDFs):**

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

> Tesseract is only needed for scanned (image-based) PDFs. Digital PDFs and DOCX files work without it.

### API keys

| Service | Where to get it | Required? |
|---------|----------------|-----------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | **Yes** — GPT-4o for extraction and Q&A |

> GPT-4o costs approximately **$0.02–0.15 per document** (element extraction + cross-document relationship extraction).

### Ports that must be free

| Port | Used by |
|------|--------|
| `5432` | PostgreSQL (workspace metadata) |
| `7474` | Neo4j browser (optional debug UI) |
| `7687` | Neo4j Bolt |
| `6333` | Qdrant REST |
| `6334` | Qdrant gRPC |
| `8000` | FastAPI backend |
| `5173` | React frontend (Vite dev server) |

---

## Quick Start

### 1 — Clone / download

```bash
cd ~/Desktop
git clone <repo-url> "GraphRAG POC"
cd "GraphRAG POC"
```

### 2 — Set your API key

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and fill in:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

Everything else has working defaults that match the docker-compose services.

### 3 — Install Tesseract (for scanned PDFs)

```bash
brew install tesseract   # macOS
# or: sudo apt-get install tesseract-ocr
```

### 4 — Start all backing services

```bash
docker compose up -d
```

Starts **Neo4j**, **Qdrant**, and **PostgreSQL**. Wait ~20 seconds, then verify:

```bash
docker compose ps   # all three should show "healthy" or "running"
```

> **Neo4j browser** is at http://localhost:7474 (user: `neo4j`, password: `password`).

### 5 — Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

> First run downloads the BGE-M3 model (~2 GB). This happens once and is cached by Hugging Face.

### 6 — Install frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 7 — Start both servers

**Terminal 1 — API:**
```bash
./start_api.sh
# → http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
./start_frontend.sh
# → http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## Configuration Reference

All settings live in `backend/.env`:

```env
# ── OpenAI ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...        # Required
LLM_MODEL=gpt-4o                  # Change to gpt-4o-mini to reduce cost

# ── Neo4j ───────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=graphrag_elements   # Base name — per-workspace: ws_{id}

# ── PostgreSQL (workspace metadata) ─────────────────────────────────────────
POSTGRES_URL=postgresql://graphrag:graphrag@localhost:5432/graphrag

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-m3       # Change to all-MiniLM-L6-v2 for faster dev
EMBEDDING_DIMENSION=1024          # Must match the model's output dimension

# ── Quality ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD=0.5
MAX_TOKENS_EXTRACTION=4000
MAX_CHUNK_CHARS=3000
CHUNK_OVERLAP_CHARS=200
```

**Development shortcut — faster and cheaper:**

```env
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

---

## Project Structure

```
GraphRAG POC/
│
├── backend/
│   ├── api/
│   │   ├── main.py                 ← FastAPI app, CORS, router registration, Postgres init
│   │   ├── deps.py                 ← Per-workspace service factory + singleton cache
│   │   └── routes/
│   │       ├── workspaces.py       ← CRUD /api/workspaces (Postgres-backed)
│   │       ├── pipeline.py         ← POST pipeline/run — SSE streaming + coordinator pattern
│   │       ├── graph.py            ← GET graph/data, subgraph, cross-doc-relationships, stats
│   │       ├── traceability.py     ← GET traceability/coverage + chain/:id
│   │       ├── chat.py             ← POST chat/ask — intent-aware Q&A
│   │       └── status.py           ← GET/POST status, reset, elements
│   │
│   ├── config/settings.py          ← All env vars as frozen dataclass
│   │
│   ├── db/postgres.py              ← Workspace CRUD (psycopg2, sync, wrapped in to_thread)
│   │
│   ├── core/
│   │   ├── models.py               ← AtomicElement, Relationship, ParsedDocument, CoverageResult
│   │   ├── interfaces.py           ← IParser, IExtractor, IGraphStore, IVectorStore (ABCs)
│   │   └── exceptions.py
│   │
│   ├── parsers/
│   │   ├── pdf_parser.py           ← Two-pass: PyMuPDF native → Tesseract OCR fallback
│   │   └── docx_parser.py
│   │
│   ├── extractors/llm_extractor.py ← Section-aware chunking → GPT-4o function calling
│   │
│   ├── graph/
│   │   ├── neo4j_store.py          ← IGraphStore; composite (id, workspace_id) constraint
│   │   │                             get_cross_document_relationships() for sidebar
│   │   ├── builder.py              ← GraphBuilder: build, assess_coverage, traceability chain
│   │   │                             is_inter_document resolved via CONTAINS edge (robust)
│   │   └── visualizer.py           ← Legacy (unused in React UI)
│   │
│   ├── vector/
│   │   ├── embedder.py             ← BGEEmbedder singleton (lazy-loads BAAI/bge-m3)
│   │   └── qdrant_store.py         ← Per-workspace collection ws_{workspace_id}
│   │
│   └── services/
│       ├── document_service.py     ← Parse + extract + cross-doc relationship extraction
│       ├── graph_service.py        ← Workspace-scoped facade over Neo4j + Qdrant
│       └── qa_service.py           ← Intent detection → evidence → GPT-4o synthesis
│
├── frontend/
│   └── src/
│       ├── main.tsx                ← BrowserRouter wrapper
│       ├── App.tsx                 ← Route definitions (React Router v7)
│       ├── types.ts                ← Shared TypeScript interfaces
│       ├── index.css               ← Tailwind + dark/light theme CSS custom properties
│       │                             Border color overrides for Tailwind v3 CSS-var issue
│       ├── api/client.ts           ← All fetch wrappers + SSE reader; all take workspaceId
│       ├── store/pipelineStore.ts  ← Zustand: byWorkspace map — workspace-scoped jobs, persists across navigation
  ├── store/globalToastStore.ts ← Cross-workspace pipeline completion toasts
│       ├── theme/ThemeContext.tsx  ← Dark/light theme provider
│       ├── pages/
│       │   ├── WorkspacesPage.tsx  ← / — workspace grid, create, delete
│       │   └── WorkspacePage.tsx   ← /workspace/:id — keep-alive tabs, URL routing
│       └── components/
│           ├── WorkflowPanel.tsx   ← Upload zone + SSE pipeline trigger + run history
│           ├── KnowledgeGraph.tsx  ← React Flow + d3-force/dagre + cross-doc sidebar
│           ├── ElementsTable.tsx   ← Filterable/sortable elements table
│           ├── TraceabilityView.tsx← 4-column card layout, INTER/INTRA badges
│           ├── ChatWindow.tsx      ← Floating chat + evidence source cards
│           ├── PipelineProgress.tsx← Step stepper UI
│           ├── UploadZone.tsx      ← Drag-and-drop file zone
│           ├── GraphRAGLogo.tsx    ← SVG logo component
│           ├── ThemeToggle.tsx     ← Dark/light mode toggle button
│           └── Toast.tsx           ← Toast notification system
│
├── Data_Samples/                   ← Sample procurement documents for testing
├── docker-compose.yml              ← Neo4j 5.x + Qdrant + PostgreSQL 16
├── start_api.sh
└── start_frontend.sh
```

---

## UI Walkthrough

### Workspace grid (`/`)

The app opens to a card grid of all workspaces (stored in PostgreSQL):

- **Create workspace** — name + optional description → isolated Neo4j/Qdrant scope
- **Open workspace** — click a card to enter that workspace
- **Delete workspace** — removes all Neo4j nodes, Qdrant collection, and Postgres row

### Workspace app (`/workspace/:id/:tab`)

Four tabs — **Ingest · Elements · Graph · Traceability**. Tabs use CSS keep-alive (absolute positioning, `display: none` when inactive) so switching tabs does not remount components or lose state. URL updates to `/workspace/:id/:tab` — deep links and browser back/forward work correctly.

#### Ingest tab

Drop PDF or DOCX files. Filename keywords control document-type detection:

| Filename keyword | Detected as |
|-----------------|-------------|
| `rfp`, `rfx`, `tender` | RFP → extracts `Requirement` nodes |
| `risk`, `rmc`, `register` | Risk Sheet → extracts `Risk` + `Mitigation` |
| `contract`, `offer`, `agreement` | Contract → extracts `Clause` + `LD` |

Click **Run Pipeline**. Five steps stream live via SSE with verbose per-chunk and per-page OCR logs. Re-uploading the same file is safe — SHA-256 dedup skips it.

The right column shows a **step stepper**, live activity log, and **Run History** (all past pipeline runs per workspace). Pipeline state is workspace-scoped — navigating away and back preserves the run history and status. If a pipeline completes while you are in a different workspace, a **global toast notification** appears with a link to navigate back.

#### Elements tab

Filterable table of all extracted elements. Filter by type pill, search by text/ID/source, sort any column, expand a row for full content and metadata.

#### Graph tab

Interactive knowledge graph powered by **React Flow + d3-force/dagre**:

| Node colour | Element type |
|------------|-------------|
| Indigo | Requirement |
| Emerald | Clause |
| Red | Risk |
| Amber | Mitigation |
| Purple | LD (Liquidated Damages) |
| Slate | Document |

Controls:
- **Click** a node → highlight connected edges + node detail panel
- **Double-click** a node → expand its 1-hop neighbourhood inline
- **Force / Hierarchy** → toggle between d3-force and dagre LR layout
- **CONTAINS** → show/hide Document→Element containment edges
- **Cross-Doc** → open the cross-document relationships sidebar

**Cross-Doc sidebar** shows every edge that crosses document boundaries, fetched from a dedicated backend endpoint. Includes relationship type filter pills, search, evidence text on expand, and click-to-highlight the edge in the graph.

#### Traceability tab

Left panel lists every Requirement with its coverage badge (Covered / Partial / Gap) and a coverage score progress bar. Click a requirement to see its lineage across four columns — **Clauses · Risks · Mitigations · LDs** — with **↔ INTER** (cross-document) and **↕ INTRA** (same-document) badges per element, and a gaps alert if risks lack mitigations or LDs.

#### Chat (floating)

Bottom-right floating button. Intent-aware Q&A using Cypher graph traversal + BGE-M3 semantic search. Each answer shows the query strategy used and collapsible source evidence cards.

---

## Extending the System

### Adding a new parser (e.g. Excel)

1. Create `backend/parsers/excel_parser.py` implementing `IParser` from `core/interfaces.py`
2. Register it in `backend/parsers/__init__.py` inside `ParserFactory._parsers`

### Adding a new element type

1. Add the value to `ElementType` in `backend/core/models.py`
2. Update the extraction prompt in `backend/extractors/llm_extractor.py`
3. Add the ID prefix to `prefix_map` in the same file
4. Add the node colour to `TYPE_CONFIG` in `frontend/src/components/KnowledgeGraph.tsx`
5. Add the colour to `TYPE_ACCENT` in `frontend/src/components/TraceabilityView.tsx`

### Adding a new Q&A intent

In `backend/services/qa_service.py`:
1. Add keyword detection in `_classify_intent()`
2. Add a `_gather_<intent>_evidence()` method
3. Map the new intent in `answer()`

### Useful debug endpoints

| URL | What it shows |
|-----|--------------|
| http://localhost:7474 | Neo4j Browser — run Cypher directly |
| http://localhost:6333/dashboard | Qdrant dashboard |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:8000/api/workspaces | List all workspaces |

**Useful Cypher (replace `<wid>` with workspace ID from Postgres):**

```cypher
// All elements in a workspace
MATCH (e:Element {workspace_id: '<wid>'})
RETURN e.id, e.type, e.section ORDER BY e.type

// All cross-document relationships
MATCH (a:Element {workspace_id: '<wid>'})-[r]->(b:Element {workspace_id: '<wid>'})
WHERE type(r) <> 'CONTAINS' AND a.document_id <> b.document_id
RETURN a.id, type(r), b.id, r.confidence ORDER BY type(r)

// Uncovered requirements
MATCH (req:Element {type: 'Requirement', workspace_id: '<wid>'})
WHERE NOT (req)<-[:COVERS]-() AND NOT (req)<-[:PARTIALLY_COVERS]-()
RETURN req.id, req.text, req.section
```

---

## Troubleshooting

### Coverage shows "Not Covered" for everything

1. Check backend logs — look for `Auto-flipped` messages indicating direction was corrected.
2. Wipe the workspace (Wipe button), then re-upload all documents in one pipeline run.
3. Verify document filenames contain the right keywords (`rfp`, `contract`, `risk`) so type detection works correctly.
4. Lower `CONFIDENCE_THRESHOLD=0.4` in `.env` and restart the API if relationships exist but scores are low.

### "Could not initialize PostgreSQL" on startup

Docker Compose wasn't started first, or Postgres isn't healthy yet:
```bash
docker compose up -d
docker compose ps   # confirm postgres shows "healthy"
```

### Scanned PDF extracts nothing

Check Tesseract: `tesseract --version`. Install with `brew install tesseract` (macOS) or `sudo apt-get install tesseract-ocr`.

### "Neo4j: Connection refused"

```bash
docker compose up -d
docker compose logs neo4j   # wait ~20 seconds after starting
```

### "OPENAI_API_KEY is not set"

```bash
cp backend/.env.example backend/.env
# open backend/.env and set OPENAI_API_KEY=sk-proj-...
```

### BGE-M3 download slow / fails

Switch to a smaller model for development:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### "No Requirements found" in Traceability

Document type is inferred from the filename. Rename the RFP to include `rfp` (e.g. `rfp_project.pdf`), wipe the workspace, then re-upload.

### Cross-doc sidebar shows 0 relationships

Wipe the workspace and re-ingest all documents together in one run. Relationships are extracted once in Step 3 by the coordinator — uploading all files together ensures the LLM sees the full combined element set when inferring cross-document links.

### Port conflicts

```bash
lsof -i :5432    # Postgres
lsof -i :8000    # FastAPI
lsof -i :7687    # Neo4j Bolt
```

Edit `docker-compose.yml` to remap ports and update `backend/.env` accordingly.

---

## Stopping the Services

```bash
# Stop containers (data preserved in Docker volumes)
docker compose stop

# Full reset — removes containers AND volumes (all workspaces deleted)
docker compose down -v
```
