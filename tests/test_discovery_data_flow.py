"""
Tests for the complete Discovery data flow:
  GitHub metadata → fetch_repo_metadata → filter_and_score → rank_repositories
  → get_ranking_summary → api_discover → Discovery table field mapping.

Verifies:
  - full_name, description, primary_language, relevance_score, relevance_pct,
    url are all present and correct in the final ranking output.
  - README is NOT fetched during discovery (only 2 API calls per repo).
  - Topics come from the main repo response (no separate topics call).
  - Cache prevents re-fetching the same URL twice.
  - Partial failure (rate-limited languages call) still produces displayable data.
  - Total failure (main call returns None) produces an error dict with full_name.
  - Relevance score is > 0 for repos that match the keyword.
  - Error repos show full_name derived from URL (never blank).
"""

import pytest
from unittest.mock import patch, call
from services.github_service import (
    fetch_repo_metadata, fetch_repos_metadata,
    _best_language, clear_metadata_cache,
)
from services.keyword_filter import filter_and_score
from services.repository_ranker import rank_repositories, get_ranking_summary


# ── Fixtures ──────────────────────────────────────────────────────────────

SPARK_MAIN = {
    "full_name": "apache/spark",
    "description": "Apache Spark — unified analytics engine for large-scale data processing",
    "language": "Scala",
    "topics": ["big-data", "java", "python", "scala", "spark"],
    "stargazers_count": 39000,
    "forks_count": 22000,
    "open_issues_count": 300,
    "size": 800000,
    "default_branch": "master",
    "clone_url": "https://github.com/apache/spark.git",
    "created_at": "2014-02-25T22:05:01Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

AIRFLOW_MAIN = {
    "full_name": "apache/airflow",
    "description": "Apache Airflow — a platform to programmatically author, schedule and monitor workflows",
    "language": "Python",
    "topics": ["airflow", "apache", "python", "workflow"],
    "stargazers_count": 35000,
    "forks_count": 14000,
    "open_issues_count": 1000,
    "size": 400000,
    "default_branch": "main",
    "clone_url": "https://github.com/apache/airflow.git",
    "created_at": "2015-04-14T20:50:08Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

ZALENIUM_MAIN = {
    "full_name": "zalando/zalenium",
    "description": "A flexible and scalable container based Selenium Grid",
    "language": "Java",
    "topics": ["docker", "grid", "java", "selenium", "testing"],
    "stargazers_count": 3000,
    "forks_count": 500,
    "open_issues_count": 50,
    "size": 20000,
    "default_branch": "master",
    "clone_url": "https://github.com/zalando/zalenium.git",
    "created_at": "2016-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z",
}

SPARK_LANGS    = {"Scala": 7_000_000, "Java": 2_000_000, "Python": 500_000, "R": 100_000}
AIRFLOW_LANGS  = {"Python": 3_000_000, "JavaScript": 200_000, "HTML": 100_000}
ZALENIUM_LANGS = {"Java": 800_000, "Shell": 50_000}

URLS = [
    "https://github.com/apache/spark",
    "https://github.com/airbnb/airflow",   # airbnb URL → redirects to apache, test with apache
    "https://github.com/zalando/zalenium",
]

REPO_DATA = {
    "/repos/apache/spark":    SPARK_MAIN,
    "/repos/apache/airflow":  AIRFLOW_MAIN,
    "/repos/zalando/zalenium": ZALENIUM_MAIN,
}
LANG_DATA = {
    "/repos/apache/spark/languages":    SPARK_LANGS,
    "/repos/apache/airflow/languages":  AIRFLOW_LANGS,
    "/repos/zalando/zalenium/languages": ZALENIUM_LANGS,
}

def _make_mock_get(fail_languages=False, fail_main=False):
    def mock_get(path, params=None, timeout=8):
        if fail_main:
            return None
        for repo_path, data in REPO_DATA.items():
            if path == repo_path:
                return data
        if fail_languages:
            return None
        for lang_path, data in LANG_DATA.items():
            if path == lang_path:
                return data
        return None
    return mock_get


# ── Field presence tests ──────────────────────────────────────────────────

REQUIRED_DISPLAY_FIELDS = {
    "full_name", "description", "primary_language", "code_languages", "url",
    "relevance_score", "relevance_pct", "rank", "error",
}

class TestDiscoveryFieldMapping:

    def setup_method(self):
        clear_metadata_cache()

    def _run_pipeline(self, url, keyword="python", fail_languages=False):
        with patch("services.github_service._get",
                   side_effect=_make_mock_get(fail_languages=fail_languages)):
            meta = fetch_repo_metadata(url)
        scored = filter_and_score([meta], keyword)
        ranked = rank_repositories(scored, top_n=10)
        summary = get_ranking_summary(ranked, keyword, 1)
        return summary["top_repos"][0]

    def test_spark_full_name_present(self):
        repo = self._run_pipeline("https://github.com/apache/spark")
        assert repo["full_name"] == "apache/spark"

    def test_spark_description_present(self):
        repo = self._run_pipeline("https://github.com/apache/spark")
        assert "Spark" in repo["description"] or "analytics" in repo["description"]

    def test_spark_language_present(self):
        repo = self._run_pipeline("https://github.com/apache/spark")
        # Python is in langs_data and higher priority than Scala
        assert repo["primary_language"] == "Python"

    def test_airflow_language_is_python(self):
        clear_metadata_cache()
        with patch("services.github_service._get",
                   side_effect=_make_mock_get()):
            meta = fetch_repo_metadata("https://github.com/apache/airflow")
        scored = filter_and_score([meta], "python")
        ranked = rank_repositories(scored, top_n=10)
        assert ranked[0]["primary_language"] == "Python"

    def test_zalenium_language_is_java(self):
        clear_metadata_cache()
        with patch("services.github_service._get",
                   side_effect=_make_mock_get()):
            meta = fetch_repo_metadata("https://github.com/zalando/zalenium")
        scored = filter_and_score([meta], "java selenium")
        ranked = rank_repositories(scored, top_n=10)
        assert ranked[0]["primary_language"] == "Java"

    def test_relevance_score_nonzero_for_matching_keyword(self):
        repo = self._run_pipeline("https://github.com/apache/spark", "python")
        assert repo["relevance_score"] > 0.0

    def test_relevance_pct_formatted(self):
        repo = self._run_pipeline("https://github.com/apache/spark", "python")
        assert repo["relevance_pct"].endswith("%")
        assert float(repo["relevance_pct"].rstrip("%")) > 0.0

    def test_rank_assigned(self):
        repo = self._run_pipeline("https://github.com/apache/spark", "python")
        assert repo["rank"] == 1

    def test_url_present(self):
        repo = self._run_pipeline("https://github.com/apache/spark", "python")
        assert "github.com/apache/spark" in repo["url"]

    def test_all_required_display_fields_present(self):
        repo = self._run_pipeline("https://github.com/apache/spark", "python")
        missing = REQUIRED_DISPLAY_FIELDS - set(repo.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_partial_failure_languages_still_shows_data(self):
        """When languages call is rate-limited, primary_language falls back to main response."""
        repo = self._run_pipeline(
            "https://github.com/apache/airflow", "python",
            fail_languages=True
        )
        # full_name and description must still be populated
        assert repo["full_name"] == "apache/airflow"
        assert repo["description"]
        # language falls back to main response
        assert repo["primary_language"] == "Python"  # from AIRFLOW_MAIN["language"]

    def test_total_failure_has_full_name_not_blank(self):
        """When main API call fails, full_name must still be derived from the URL."""
        clear_metadata_cache()
        with patch("services.github_service._get", return_value=None):
            meta = fetch_repo_metadata("https://github.com/apache/spark")
        assert meta["full_name"] == "apache/spark"
        assert meta["error"] is not None

    def test_error_repos_show_in_summary_not_blank(self):
        """Error repos must have full_name populated so table never shows blank."""
        clear_metadata_cache()
        with patch("services.github_service._get", return_value=None):
            meta = fetch_repo_metadata("https://github.com/apache/spark")
        scored = filter_and_score([meta], "python")
        ranked = rank_repositories(scored, top_n=10)
        summary = get_ranking_summary(ranked, "python", 1)
        repo = summary["top_repos"][0]
        assert repo["full_name"] == "apache/spark"   # never blank


# ── API call count tests ───────────────────────────────────────────────────

class TestApiCallCount:

    def setup_method(self):
        clear_metadata_cache()

    def test_discovery_uses_at_most_2_calls_per_repo(self):
        """
        fetch_repo_metadata must use at most 2 API calls during discovery
        (main + languages).  Topics come from main response; README not fetched.
        """
        calls = []

        def counting_get(path, params=None, timeout=8):
            calls.append(path)
            for rp, d in REPO_DATA.items():
                if path == rp:
                    return d
            for lp, d in LANG_DATA.items():
                if path == lp:
                    return d
            return None

        with patch("services.github_service._get", side_effect=counting_get):
            fetch_repo_metadata("https://github.com/apache/spark")

        assert len(calls) <= 2, f"Expected ≤2 API calls, got {len(calls)}: {calls}"
        readme_calls = [c for c in calls if "readme" in c.lower()]
        assert not readme_calls, f"README should not be fetched during discovery: {readme_calls}"
        topics_calls = [c for c in calls if "/topics" in c]
        assert not topics_calls, f"Topics endpoint should not be called: {topics_calls}"

    def test_cache_prevents_second_fetch(self):
        """Calling fetch_repo_metadata twice for the same URL uses 0 extra calls."""
        calls = []

        def counting_get(path, params=None, timeout=8):
            calls.append(path)
            for rp, d in REPO_DATA.items():
                if path == rp:
                    return d
            for lp, d in LANG_DATA.items():
                if path == lp:
                    return d
            return None

        with patch("services.github_service._get", side_effect=counting_get):
            fetch_repo_metadata("https://github.com/apache/spark")
            first_count = len(calls)
            fetch_repo_metadata("https://github.com/apache/spark")
            second_count = len(calls)

        assert second_count == first_count, "Cache should prevent additional API calls"

    def test_three_repos_use_at_most_6_api_calls(self):
        """3 repos × 2 calls = 6 total — well within the 60/hr limit."""
        calls = []

        def counting_get(path, params=None, timeout=8):
            calls.append(path)
            for rp, d in REPO_DATA.items():
                if path == rp:
                    return d
            for lp, d in LANG_DATA.items():
                if path == lp:
                    return d
            return None

        test_urls = [
            "https://github.com/apache/spark",
            "https://github.com/apache/airflow",
            "https://github.com/zalando/zalenium",
        ]
        with patch("services.github_service._get", side_effect=counting_get):
            fetch_repos_metadata(test_urls, max_workers=3)

        assert len(calls) <= 6, f"Expected ≤6 API calls for 3 repos, got {len(calls)}: {calls}"


# ── End-to-end summary structure ─────────────────────────────────────────

class TestSummaryStructure:

    def setup_method(self):
        clear_metadata_cache()

    def _make_ranked_summary(self, keyword="python"):
        with patch("services.github_service._get",
                   side_effect=_make_mock_get()):
            metas = fetch_repos_metadata([
                "https://github.com/apache/spark",
                "https://github.com/apache/airflow",
                "https://github.com/zalando/zalenium",
            ], max_workers=3)
        scored  = filter_and_score(metas, keyword)
        ranked  = rank_repositories(scored, top_n=10)
        summary = get_ranking_summary(ranked, keyword, len(metas))
        return summary

    def test_summary_has_top_repos(self):
        summary = self._make_ranked_summary("python")
        assert "top_repos" in summary
        assert len(summary["top_repos"]) == 3

    def test_top_repos_have_all_display_fields(self):
        summary = self._make_ranked_summary("python")
        for repo in summary["top_repos"]:
            missing = REQUIRED_DISPLAY_FIELDS - set(repo.keys())
            assert not missing, f"{repo.get('full_name')} missing: {missing}"

    def test_python_keyword_ranks_python_repos_higher(self):
        summary = self._make_ranked_summary("python")
        names = [r["full_name"] for r in summary["top_repos"]]
        # airflow (Python) should rank higher than zalenium (Java) for python keyword
        if "apache/airflow" in names and "zalando/zalenium" in names:
            af_rank = names.index("apache/airflow")
            zal_rank = names.index("zalando/zalenium")
            assert af_rank < zal_rank

    def test_keywords_field_preserved_in_summary(self):
        summary = self._make_ranked_summary("python big-data")
        assert summary["keywords"] == "python big-data"

    def test_total_submitted_count(self):
        summary = self._make_ranked_summary("python")
        assert summary["total_submitted"] == 3
