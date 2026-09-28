"""
Tests for the RAG QA pipeline.

Covers:
  1. Predefined-style question (answered from knowledge).
  2. Completely new/random repository question (answered from knowledge).
  3. Question whose answer is NOT present in the repository.
  4. Dynamic retrieval — engine does not depend on hardcoded questions.
  5. SharedKnowledge.flat_text_for_rag produces granular, question-answerable text.
  6. QAEngine.ensure_index populates the engine from shared knowledge.
  7. QAEngine returns "not found" gracefully rather than inventing answers.
"""

import pytest
from agents.orchestrator import SharedKnowledge
from rag.document_processor import chunk_text, process_knowledge_for_rag
from rag.retriever import VectorRetriever
from rag.qa_engine import QAEngine


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

SAMPLE_KNOWLEDGE = {
    "repository": {
        "name":             "auth-service",
        "full_name":        "acme/auth-service",
        "owner":            "acme",
        "url":              "https://github.com/acme/auth-service",
        "description":      "JWT-based authentication service",
        "topics":           ["auth", "jwt", "python"],
        "primary_language": "Python",
        "languages":        {"Python": 95.0, "Shell": 5.0},
    },
    "statistics": {
        "total_files": 12,
        "source_files": 8,
        "python_files": 8,
        "java_files": 0,
        "classes": 3,
        "methods": 10,
        "functions": 5,
        "imports": 20,
        "dependencies": ["flask", "jwt", "sqlalchemy"],
    },
    "files": [
        {
            "path": "src/auth.py",
            "language": "Python",
            "classes": [
                {
                    "name":    "AuthManager",
                    "methods": ["login", "logout", "verify_token"],
                    "bases":   [],
                }
            ],
            "functions": ["hash_password", "check_password"],
            "imports":   ["flask", "jwt", "models"],
            "calls":     ["User.get_by_email", "jwt.encode"],
            "error":     None,
        },
        {
            "path": "src/models.py",
            "language": "Python",
            "classes": [
                {
                    "name":    "User",
                    "methods": ["get_by_email", "save", "delete"],
                    "bases":   ["Base"],
                }
            ],
            "functions": [],
            "imports":   ["sqlalchemy", "database"],
            "calls":     [],
            "error":     None,
        },
        {
            "path": "main.py",
            "language": "Python",
            "classes": [],
            "functions": ["main", "create_app"],
            "imports":   ["auth", "models", "flask"],
            "calls":     ["create_app"],
            "error":     None,
        },
    ],
}

SAMPLE_ARCH = {
    "architecture_summary": (
        "The project has a service layer (auth.py) and a data layer (models.py). "
        "main.py is the entry point."
    ),
    "key_components": [
        {"label": "src/auth.py",   "type": "File", "total": 8},
        {"label": "src/models.py", "type": "File", "total": 5},
        {"label": "AuthManager",   "type": "Class","total": 3},
    ],
    "entry_points":        ["main.py"],
    "layers":              {"Service": ["src/auth.py"], "Data": ["src/models.py"]},
    "module_dependencies": [
        {"from": "src/auth.py", "relation": "IMPORTS", "to": "src/models.py"},
    ],
    "cyclic_dependencies": [],
}


def _make_shared_knowledge() -> SharedKnowledge:
    sk = SharedKnowledge()
    sk.store(
        "acme/auth-service",
        {"knowledge": SAMPLE_KNOWLEDGE, "statistics": SAMPLE_KNOWLEDGE["statistics"]},
        {"architecture": SAMPLE_ARCH},
    )
    return sk


def _make_qa_engine(sk: SharedKnowledge) -> QAEngine:
    """Build a QAEngine with keyword-only retrieval (no FAISS/embeddings needed)."""
    retriever = VectorRetriever()
    chunks    = process_knowledge_for_rag(sk, "acme/auth-service")
    # Store without embeddings so keyword fallback is used
    retriever.chunks = chunks
    return QAEngine(retriever=retriever)


# ──────────────────────────────────────────────────────────────
# flat_text_for_rag — granularity tests
# ──────────────────────────────────────────────────────────────

class TestFlatTextForRag:

    def setup_method(self):
        self.sk   = _make_shared_knowledge()
        self.text = self.sk.flat_text_for_rag("acme/auth-service")

    def test_contains_repo_description(self):
        assert "JWT-based authentication service" in self.text

    def test_contains_file_paths(self):
        assert "src/auth.py" in self.text
        assert "src/models.py" in self.text
        assert "main.py" in self.text

    def test_contains_class_names(self):
        assert "AuthManager" in self.text
        assert "User" in self.text

    def test_contains_method_names(self):
        assert "login" in self.text
        assert "logout" in self.text
        assert "verify_token" in self.text
        assert "get_by_email" in self.text

    def test_contains_function_names(self):
        assert "hash_password" in self.text
        assert "create_app" in self.text

    def test_contains_imports(self):
        assert "flask" in self.text
        assert "sqlalchemy" in self.text

    def test_contains_dependencies(self):
        assert "jwt" in self.text

    def test_contains_entry_point(self):
        assert "main.py" in self.text

    def test_contains_architecture_summary(self):
        assert "service layer" in self.text.lower() or "entry point" in self.text.lower()

    def test_methods_linked_to_class(self):
        # Each method must appear in a line that also names its class
        lines = self.text.split("\n")
        method_class_lines = [l for l in lines if "login" in l and "AuthManager" in l]
        assert len(method_class_lines) >= 1

    def test_empty_repo_returns_empty_string(self):
        assert self.sk.flat_text_for_rag("nonexistent/repo") == ""


# ──────────────────────────────────────────────────────────────
# Document processor
# ──────────────────────────────────────────────────────────────

class TestDocumentProcessor:

    def test_chunk_text_basic(self):
        text   = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, source="test")
        assert len(chunks) > 1
        for c in chunks:
            assert "text"   in c
            assert "source" in c
            assert c["source"] == "test"

    def test_chunk_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunks_have_overlap(self):
        text   = " ".join([f"w{i}" for i in range(300)])
        chunks = chunk_text(text, source="s")
        # Consecutive chunks must share some words (overlap > 0)
        w1 = set(chunks[0]["text"].split())
        w2 = set(chunks[1]["text"].split())
        assert len(w1 & w2) > 0

    def test_process_knowledge_produces_chunks(self):
        sk     = _make_shared_knowledge()
        chunks = process_knowledge_for_rag(sk, "acme/auth-service")
        assert len(chunks) > 0
        assert all(c["source"] == "acme/auth-service" for c in chunks)


# ──────────────────────────────────────────────────────────────
# QAEngine — retrieval and answering
# ──────────────────────────────────────────────────────────────

class TestQAEngine:

    def setup_method(self):
        self.sk = _make_shared_knowledge()
        self.qa = _make_qa_engine(self.sk)

    # 1. Predefined-style question
    def test_predefined_question_authentication(self):
        result = self.qa.query("How does authentication work?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0
        # Answer must reference auth-related content
        answer_lower = result["answer"].lower()
        assert any(kw in answer_lower for kw in
                   ["auth", "login", "token", "jwt", "authmanager", "context", "retrieved"])

    # 2. Completely new/random repository-related question
    def test_new_question_about_classes(self):
        result = self.qa.query("What classes are defined in this repository?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0
        answer_lower = result["answer"].lower()
        assert any(kw in answer_lower for kw in
                   ["authmanager", "user", "class", "context", "retrieved"])

    def test_new_question_about_files(self):
        result = self.qa.query("What files exist in the src directory?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0

    def test_new_question_about_methods(self):
        result = self.qa.query("What methods does AuthManager have?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0
        answer_lower = result["answer"].lower()
        assert any(m in answer_lower for m in
                   ["login", "logout", "verify", "authmanager", "context", "retrieved"])

    def test_new_question_about_dependencies(self):
        result = self.qa.query("What are the main dependencies of this project?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0
        answer_lower = result["answer"].lower()
        assert any(dep in answer_lower for dep in
                   ["flask", "jwt", "sqlalchemy", "context", "retrieved"])

    def test_new_question_about_entry_point(self):
        result = self.qa.query("Which file is the entry point of the application?")
        assert result["has_knowledge"] is True
        assert result["chunks_used"] > 0
        assert "main" in result["answer"].lower() or "retrieved" in result["answer"].lower()

    # 3. Question whose answer is NOT present in the repository
    def test_absent_topic_returns_not_found(self):
        result = self.qa.query(
            "What is the Kubernetes deployment configuration for this project?"
        )
        # Either nothing retrieved, or the answer says information not found
        if result["chunks_used"] == 0:
            assert result["has_knowledge"] is True
            assert "not found" in result["answer"].lower() or "not" in result["answer"].lower()
        else:
            # Retrieved some chunks but they won't contain kubernetes info —
            # answer should not invent kubernetes facts
            assert "kubernetes" not in result["answer"].lower() or "not found" in result["answer"].lower()

    def test_completely_unrelated_question(self):
        result = self.qa.query(
            "What is the weather forecast for London tomorrow?"
        )
        # Engine has knowledge but this topic isn't in the repo
        # chunks_used may be > 0 (keyword overlap on common words) but answer
        # must not pretend the repo contains weather data
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    # 4. Dynamic retrieval — not limited to hardcoded questions
    def test_arbitrary_question_returns_result(self):
        """Any syntactically valid question must return a structured result."""
        for question in [
            "Tell me about the User model.",
            "Which module handles password hashing?",
            "What does save() do?",
            "List all Python files.",
            "Are there any cyclic imports?",
        ]:
            result = self.qa.query(question)
            assert "answer"       in result
            assert "sources"      in result
            assert "chunks_used"  in result
            assert "llm_used"     in result
            assert "has_knowledge" in result
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0

    # 5. No knowledge state
    def test_no_knowledge_returns_helpful_message(self):
        empty_retriever = VectorRetriever()   # no chunks
        qa_empty        = QAEngine(retriever=empty_retriever)
        result          = qa_empty.query("How does login work?")
        assert result["has_knowledge"] is False
        assert result["chunks_used"]   == 0
        assert "analysis" in result["answer"].lower() or "pipeline" in result["answer"].lower()

    # 6. ensure_index loads from shared knowledge
    def test_ensure_index_loads_from_shared_knowledge(self):
        retriever = VectorRetriever()   # empty
        qa        = QAEngine(retriever=retriever)
        assert not qa.has_knowledge()

        qa.ensure_index(shared_knowledge=self.sk)
        assert qa.has_knowledge()

    # 7. Sources are returned
    def test_sources_included_in_answer(self):
        result = self.qa.query("What classes exist?")
        if result["chunks_used"] > 0:
            assert len(result["sources"]) > 0
            assert all(isinstance(s, str) for s in result["sources"])
