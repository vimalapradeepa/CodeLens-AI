# CodeLens AI

**AI-Powered Multi-Agent Open Source Project Analyzer**

Final Year Project — AI & Data Science

---

## 1. Project Overview

CodeLens AI is a software comprehension platform that accepts multiple GitHub repository URLs, filters them by keyword relevance, ranks them, and then dispatches two independent custom AI agents to deeply analyze repository structure and architecture. A RAG-powered assistant answers developer questions grounded in the actual source code.

---

## 2. Motivation & Problem Statement

Developers and researchers frequently need to evaluate and understand unfamiliar open-source repositories. Manually reading thousands of files is impractical. CodeLens AI automates this process through multi-agent AI: one agent extracts code knowledge, a second builds a semantic relationship graph, and a RAG system enables natural-language Q&A over the combined knowledge.

---

## 3. Objectives

- Accept 5–10 GitHub repository URLs with keyword filtering
- Rank repositories by a transparent relevance scoring algorithm
- Extract code entities (classes, methods, functions, imports) from Python and Java
- Build a semantic code relationship graph
- Infer architecture layers and key components
- Answer developer questions with source-file references (RAG)

---

## 4. Architecture

```
                 CodeLens AI
                      |
                      ▼
           Multi-Agent Orchestrator
                      |
         ┌────────────┴────────────┐
         ▼                         ▼
     Agent 1                   Agent 2
Repository Analysis     Semantic Code Graph
         |                         |
  Repository Knowledge       Graph / Architecture
         └────────────┬────────────┘
                      ▼
               Shared Knowledge
                      ▼
                     RAG
                      ▼
                     LLM
                      ▼
            Developer Assistant
```

---

## 5. Phase 1 — Repository Analysis Agent

**Agent 1: Repository Analysis Agent** (`agents/repository_agent.py`)

Custom agent built from scratch. Responsibilities:
- GitHub URL validation (5–10 repos, no duplicates)
- Repository metadata collection via GitHub REST API
- Keyword-based relevance scoring with configurable weights
- Top-10 repository ranking
- Shallow repository cloning
- Python AST-based code analysis (classes, methods, functions, imports, calls)
- Java source analysis (javalang + regex fallback)
- Repository statistics computation
- Structured JSON knowledge generation
- Human-readable Markdown report generation

### Relevance Scoring Weights (configurable in `config.py`)

| Field | Weight |
|-------|--------|
| Repository name | 25% |
| Description | 20% |
| Topics | 20% |
| README | 15% |
| Code entities | 12% |
| Language | 8% |

---

## 6. Phase 2 — Semantic Code Graph Agent + RAG

**Agent 2: Semantic Code Graph Agent** (`agents/architecture_agent.py`)

Custom agent built from scratch. Responsibilities:
- Consume Agent 1 structured knowledge
- Build semantic code relationship graph (NetworkX)
- Node types: Repository, File, Class, Method, Function, Module, Package
- Edge types: CONTAINS, IMPORTS, CALLS, INHERITS, DEFINES, DEPENDS_ON
- Architecture layer inference (Presentation / Service / Data)
- Entry-point detection
- Key component identification (graph centrality)
- Cyclic dependency detection
- Interactive HTML visualisation (PyVis)

### RAG Pipeline (`rag/`)

```
Agent 1 Knowledge + Agent 2 Graph
          ↓
     Chunking (500 tokens, 50 overlap)
          ↓
  sentence-transformers embeddings (local)
          ↓
       FAISS vector store
          ↓
    Top-5 chunk retrieval
          ↓
        LLM prompt
          ↓
   Developer answer + source files
```

---

## 7. Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| Code analysis | Python `ast`, `javalang` |
| Graph | NetworkX 3.3, PyVis 0.3 |
| RAG embeddings | sentence-transformers |
| Vector store | FAISS |
| LLM | OpenAI API / Groq (configurable) |
| Frontend | HTML5, CSS3, Vanilla JS |
| GitHub data | GitHub REST API v3 |
| VCS | Git (shallow clone) |

---

## 8. Installation

### Prerequisites
- Python 3.10+
- Git installed and on PATH

### Setup

```bash
git clone <your-repo-url>
cd CodeLens-AI

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add GITHUB_TOKEN and optionally LLM_API_KEY
```

---

## 9. Running the Application

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 10. Testing

```bash
python -m pytest tests/ -v
```

**51 tests** covering:
- URL validation
- Duplicate/limit enforcement
- Keyword filtering & relevance scoring
- Repository ranking
- Language detection
- Python AST analysis
- Java source analysis
- Repository scanner
- Statistics computation
- Agent 1 pipeline (mocked)
- Agent 2 graph construction
- Architecture analysis

---

## 11. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub PAT (increases rate limit 60→5000/hr) | — |
| `LLM_PROVIDER` | `openai` \| `groq` | `openai` |
| `LLM_API_KEY` | LLM provider API key | — |
| `LLM_MODEL` | Model name | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |

The application works **without any API keys**: code analysis, graph generation, and keyword-based retrieval all function locally. LLM answers require `LLM_API_KEY`.

---

## 12. Limitations

- Clone of very large repos (>500 MB) may be slow
- Java analysis accuracy depends on `javalang` parser compatibility
- RAG quality improves with an LLM API key; without one, raw retrieved context is shown
- Graph visualisation performance degrades for repos with >5000 nodes

---

## 13. Future Work

- Support for JavaScript/TypeScript, C++, Go
- Incremental analysis (re-analyse only changed files)
- Cross-repository relationship analysis
- GitHub Actions integration
- Fine-tuned code LLM for richer answers
- Database persistence (PostgreSQL) instead of JSON files
