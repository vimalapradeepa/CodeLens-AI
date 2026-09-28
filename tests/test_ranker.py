"""Tests for keyword filtering and ranking."""
import pytest
from services.keyword_filter import score_repository, filter_and_score
from services.repository_ranker import rank_repositories


SAMPLE_REPOS = [
    {
        "name": "ml-toolkit",
        "description": "Machine learning toolkit with RAG support",
        "topics": ["machine-learning", "rag", "nlp"],
        "readme": "A comprehensive machine learning library for NLP tasks",
        "primary_language": "Python",
        "error": None,
    },
    {
        "name": "web-framework",
        "description": "Fast web framework for Python",
        "topics": ["web", "flask", "api"],
        "readme": "Build web APIs quickly",
        "primary_language": "Python",
        "error": None,
    },
    {
        "name": "data-science-utils",
        "description": "Utilities for data science and NLP",
        "topics": ["data-science", "nlp", "pandas"],
        "readme": "Helper functions for ML and NLP processing",
        "primary_language": "Python",
        "error": None,
    },
    {
        "name": "java-orm",
        "description": "Java ORM framework for databases",
        "topics": ["java", "database", "orm"],
        "readme": "Object relational mapping in Java",
        "primary_language": "Java",
        "error": None,
    },
    {
        "name": "neural-net",
        "description": "Neural network and deep learning RAG system",
        "topics": ["deep-learning", "rag", "transformer"],
        "readme": "Build neural networks and retrieval augmented generation",
        "primary_language": "Python",
        "error": None,
    },
]


class TestKeywordFilter:

    def test_relevant_score_higher(self):
        score_ml  = score_repository(SAMPLE_REPOS[0], "machine learning NLP")
        score_web = score_repository(SAMPLE_REPOS[1], "machine learning NLP")
        assert score_ml > score_web

    def test_no_keywords_neutral(self):
        score = score_repository(SAMPLE_REPOS[0], "")
        assert score == 50.0

    def test_topic_match(self):
        score = score_repository(SAMPLE_REPOS[0], "rag")
        assert score > 0

    def test_filter_and_score_sorted(self):
        scored = filter_and_score(SAMPLE_REPOS, "machine learning RAG NLP")
        scores = [r["relevance_score"] for r in scored]
        assert scores == sorted(scores, reverse=True)

    def test_error_repo_score_zero(self):
        bad = {"name": "x", "error": "not found", "description": "", "topics": [], "readme": "", "primary_language": ""}
        result = filter_and_score([bad], "machine learning")
        assert result[0]["relevance_score"] == 0.0

    def test_language_preserved_after_scoring(self):
        """primary_language must never be mutated or removed by filter_and_score."""
        import copy
        repos = copy.deepcopy(SAMPLE_REPOS)
        scored = filter_and_score(repos, "java python")
        for r in scored:
            # Every repo that originally had a primary_language must still have it
            original = next(o for o in SAMPLE_REPOS if o["name"] == r["name"])
            assert r.get("primary_language") == original["primary_language"]

    def test_language_keyword_scores_matching_language_higher(self):
        """Keywords containing a language name should rank that language's repo higher."""
        scored = filter_and_score(SAMPLE_REPOS, "java")
        java_repo = next(r for r in scored if r["name"] == "java-orm")
        python_repo = next(r for r in scored if r["name"] == "web-framework")
        assert java_repo["relevance_score"] > python_repo["relevance_score"]

    def test_none_primary_language_handled(self):
        """A repo with primary_language=None must not crash and must score 0 for language field."""
        repo = {
            "name": "no-lang-repo",
            "description": "A repository with no detected language",
            "topics": [],
            "readme": "",
            "primary_language": None,
            "error": None,
        }
        score = score_repository(repo, "python java")
        assert isinstance(score, float)  # must not raise


class TestRanker:

    def test_top_n(self):
        scored = filter_and_score(SAMPLE_REPOS, "RAG NLP")
        ranked = rank_repositories(scored, top_n=3)
        assert len(ranked) == 3
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_rank_order(self):
        scored = filter_and_score(SAMPLE_REPOS, "machine learning NLP RAG")
        ranked = rank_repositories(scored, top_n=5)
        scores = [r["relevance_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_relevance_pct_format(self):
        scored = filter_and_score(SAMPLE_REPOS, "RAG")
        ranked = rank_repositories(scored, top_n=2)
        for r in ranked:
            assert r["relevance_pct"].endswith("%")
