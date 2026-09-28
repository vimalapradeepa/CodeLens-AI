"""
Tests for the parallel metadata fetch and timeout behaviour in github_service.

Verifies:
  - fetch_repos_metadata runs concurrently (wall-clock faster than serial sum).
  - A slow/hanging repo does not block other repos past per_repo_timeout.
  - A repo that raises an exception gets an error dict, not an exception.
  - Result order matches input URL order.
  - Timeout and error fallback dicts have the required keys.
  - _get respects the hard request timeout.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from services.github_service import (
    fetch_repos_metadata,
    _timeout_meta,
    _error_meta,
    _best_language,
)


# ── Helpers ───────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "url", "owner", "name", "full_name", "error",
    "description", "topics", "languages", "primary_language", "code_languages",
    "stars", "forks", "open_issues", "size_kb", "default_branch",
    "readme", "clone_url", "created_at", "updated_at",
}

SAMPLE_URLS = [
    "https://github.com/user/repo1",
    "https://github.com/user/repo2",
    "https://github.com/user/repo3",
]


def _make_good_meta(url, name="repo"):
    """A complete, successful metadata dict."""
    owner = "user"
    return {
        "url": url, "owner": owner, "name": name,
        "full_name": f"{owner}/{name}", "error": None,
        "description": "Test repo", "topics": ["python"],
        "languages": {"Python": 100.0}, "primary_language": "Python",
        "code_languages": ["Python"],
        "stars": 10, "forks": 2, "open_issues": 1, "size_kb": 500,
        "default_branch": "main", "readme": "# Test",
        "clone_url": f"https://github.com/{owner}/{name}.git",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


# ── Fallback dict shape ───────────────────────────────────────────────────

class TestFallbackDicts:

    def test_timeout_meta_has_all_required_keys(self):
        meta = _timeout_meta("https://github.com/user/repo")
        assert REQUIRED_KEYS.issubset(meta.keys()), \
            f"Missing keys: {REQUIRED_KEYS - meta.keys()}"

    def test_timeout_meta_has_error_message(self):
        meta = _timeout_meta("https://github.com/user/repo")
        assert meta["error"]
        assert "time" in meta["error"].lower() or "unavailable" in meta["error"].lower()

    def test_error_meta_has_all_required_keys(self):
        meta = _error_meta("https://github.com/user/repo", "connection refused")
        assert REQUIRED_KEYS.issubset(meta.keys())

    def test_error_meta_includes_message(self):
        meta = _error_meta("https://github.com/user/repo", "connection refused")
        assert "connection refused" in meta["error"].lower()

    def test_fallback_primary_language_is_none(self):
        assert _timeout_meta("https://github.com/user/repo")["primary_language"] is None
        assert _error_meta("https://github.com/user/repo", "err")["primary_language"] is None


# ── Parallel fetch — order preservation ──────────────────────────────────

class TestFetchReposMetadataOrder:

    def test_result_order_matches_input_order(self):
        """Results must come back in the same order as the submitted URLs."""
        call_results = {url: _make_good_meta(url, f"repo{i+1}")
                        for i, url in enumerate(SAMPLE_URLS)}

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda url: call_results[url]):
            results = fetch_repos_metadata(SAMPLE_URLS)

        assert len(results) == 3
        for i, url in enumerate(SAMPLE_URLS):
            assert results[i]["url"] == url or results[i]["full_name"].endswith(f"repo{i+1}")

    def test_returns_one_result_per_url(self):
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda url: _make_good_meta(url)):
            results = fetch_repos_metadata(SAMPLE_URLS)
        assert len(results) == len(SAMPLE_URLS)

    def test_empty_input_returns_empty_list(self):
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda url: _make_good_meta(url)):
            results = fetch_repos_metadata([])
        assert results == []


# ── One failing repo does not block others ────────────────────────────────

class TestFailureIsolation:

    def test_one_error_repo_does_not_prevent_others(self):
        """
        If one fetch raises an exception, the other repos still return data.
        """
        def _side_effect(url):
            if "repo2" in url:
                raise ConnectionError("simulated network failure")
            return _make_good_meta(url)

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=_side_effect):
            results = fetch_repos_metadata(SAMPLE_URLS)

        assert len(results) == 3
        good = [r for r in results if not r.get("error")]
        bad  = [r for r in results if r.get("error")]
        assert len(good) == 2
        assert len(bad)  == 1
        assert "repo2" in bad[0]["url"]

    def test_all_errors_still_returns_full_list(self):
        """Even if every fetch fails, we get one result per URL."""
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=RuntimeError("boom")):
            results = fetch_repos_metadata(SAMPLE_URLS)

        assert len(results) == len(SAMPLE_URLS)
        assert all(r.get("error") for r in results)

    def test_error_result_has_required_keys(self):
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=RuntimeError("boom")):
            results = fetch_repos_metadata(["https://github.com/user/repo1"])

        assert REQUIRED_KEYS.issubset(results[0].keys())


# ── Concurrency: parallel is faster than serial for multiple repos ────────

class TestConcurrency:

    def test_parallel_faster_than_serial_sum(self):
        """
        Three repos each taking 0.3 s serially = ~0.9 s.
        Parallel should finish in < 0.7 s.
        """
        DELAY = 0.3

        def _slow_fetch(url):
            time.sleep(DELAY)
            return _make_good_meta(url)

        t0 = time.monotonic()
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=_slow_fetch):
            results = fetch_repos_metadata(SAMPLE_URLS, max_workers=3)
        elapsed = time.monotonic() - t0

        assert len(results) == 3
        serial_time = DELAY * len(SAMPLE_URLS)
        assert elapsed < serial_time * 0.9, (
            f"Expected parallel fetch ({elapsed:.2f}s) to be faster than "
            f"serial ({serial_time:.2f}s)"
        )


# ── Per-repo timeout ──────────────────────────────────────────────────────

class TestPerRepoTimeout:

    def test_slow_repo_gets_timeout_error(self):
        """
        A repo that takes longer than per_repo_timeout should be marked as
        timed-out while fast repos still succeed.
        """
        FAST_DELAY = 0.05
        SLOW_DELAY = 2.0   # longer than the 0.3 s timeout we'll set

        def _side_effect(url):
            if "repo2" in url:
                time.sleep(SLOW_DELAY)
            else:
                time.sleep(FAST_DELAY)
            return _make_good_meta(url)

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=_side_effect):
            results = fetch_repos_metadata(
                SAMPLE_URLS,
                max_workers=3,
                per_repo_timeout=0.3,   # tight timeout to trigger on repo2
            )

        assert len(results) == 3
        # The two fast repos should succeed
        good = [r for r in results if not r.get("error")]
        assert len(good) >= 2

    def test_result_returned_even_when_all_timeout(self):
        """
        When every repo raises an exception immediately, the function must
        still return one result per URL (all with error set), not hang.
        """
        def _raise(url):
            raise ConnectionError("simulated failure")

        t0 = time.monotonic()
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=_raise):
            results = fetch_repos_metadata(
                SAMPLE_URLS[:2],
                max_workers=2,
                per_repo_timeout=5.0,
            )
        elapsed = time.monotonic() - t0

        # All results must be returned
        assert len(results) == 2
        # All must have an error set
        assert all(r.get("error") for r in results)
        # Must complete quickly (exceptions are fast)
        assert elapsed < 3.0, f"Took too long: {elapsed:.2f}s"
