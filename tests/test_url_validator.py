"""Tests for URL validation."""
import pytest
from services.url_validator import (
    validate_github_url, normalize_url, parse_owner_repo, validate_url_list
)


class TestValidateGithubUrl:
    def test_valid(self):
        ok, err = validate_github_url("https://github.com/owner/repo")
        assert ok and err == ""

    def test_valid_git_suffix(self):
        ok, _ = validate_github_url("https://github.com/owner/repo.git")
        assert ok

    def test_invalid_domain(self):
        ok, err = validate_github_url("https://gitlab.com/owner/repo")
        assert not ok

    def test_empty(self):
        ok, err = validate_github_url("")
        assert not ok

    def test_no_repo(self):
        ok, _ = validate_github_url("https://github.com/owner")
        assert not ok


class TestNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert normalize_url("https://github.com/a/b/") == "https://github.com/a/b"

    def test_strips_git(self):
        assert normalize_url("https://github.com/a/b.git") == "https://github.com/a/b"


class TestParseOwnerRepo:
    def test_parse(self):
        owner, repo = parse_owner_repo("https://github.com/openai/gpt-2")
        assert owner == "openai"
        assert repo  == "gpt-2"


class TestValidateUrlList:
    BASE = [f"https://github.com/user/repo{i}" for i in range(1, 6)]

    def test_valid_5(self):
        r = validate_url_list(self.BASE)
        assert r["count"] == 5
        assert not any("at least" in e["message"] for e in r["errors"])

    def test_valid_1(self):
        r = validate_url_list(["https://github.com/user/repo1"])
        assert r["count"] == 1
        assert not any("at least" in e["message"] for e in r["errors"])

    def test_valid_3(self):
        r = validate_url_list(self.BASE[:3])
        assert r["count"] == 3
        assert not any("at least" in e["message"] for e in r["errors"])

    def test_empty_list_rejected(self):
        r = validate_url_list([])
        assert r["count"] == 0
        assert any("at least" in e["message"] for e in r["errors"])

    # ── Duplicate handling ────────────────────────────────────

    def test_single_duplicate_removed(self):
        """6 entries with 1 duplicate → 5 unique repos, 1 duplicate recorded."""
        urls = self.BASE + [self.BASE[0]]          # 6 entries, 1 dup
        r = validate_url_list(urls)
        assert r["count"] == 5
        assert len(r["duplicates"]) == 1
        assert self.BASE[0] in r["duplicates"]

    def test_multiple_duplicates_removed(self):
        """6 entries with 3 duplicates → 3 unique repos, 3 duplicates recorded."""
        base3 = self.BASE[:3]
        urls  = base3 + base3                      # [r1,r2,r3, r1,r2,r3]
        r = validate_url_list(urls)
        assert r["count"] == 3
        assert len(r["duplicates"]) == 3

    def test_duplicate_count_matches_example(self):
        """Spec example: 6 entries, 3 duplicates → 4 unique."""
        # 4 distinct + 2 extra copies of the first two
        base4 = [f"https://github.com/user/repo{i}" for i in range(1, 5)]
        extras = base4[:2]                         # 2 duplicates of repo1 and repo2
        urls   = base4 + extras                    # 6 entered, 2 dupes
        r = validate_url_list(urls)
        assert r["count"] == 4
        assert len(r["duplicates"]) == 2

    def test_duplicate_not_in_valid(self):
        """Duplicate URL must never appear in the valid list."""
        urls = self.BASE + [self.BASE[2], self.BASE[2]]   # 2 extra copies of repo3
        r = validate_url_list(urls)
        assert r["valid"].count(normalize_url(self.BASE[2])) == 1

    def test_duplicate_produces_warning_not_error(self):
        """Duplicates go into warnings, not hard errors."""
        urls = self.BASE + [self.BASE[0]]
        r = validate_url_list(urls)
        assert len(r["warnings"]) == 1
        assert len(r["duplicates"]) == 1
        # Hard errors list must NOT contain the duplicate URL
        dup_url = normalize_url(self.BASE[0])
        hard_error_urls = [normalize_url(e["url"]) for e in r["errors"] if e["url"]]
        assert dup_url not in hard_error_urls

    def test_git_suffix_treated_as_duplicate(self):
        """.git and non-.git variants of the same URL count as duplicates."""
        urls = [
            "https://github.com/user/repo1",
            "https://github.com/user/repo1.git",   # same repo
        ] + self.BASE[1:]
        r = validate_url_list(urls)
        assert r["count"] == 5      # repo1 counted once
        assert len(r["duplicates"]) == 1

    def test_trailing_slash_treated_as_duplicate(self):
        """Trailing slash variant is treated as duplicate of the clean URL."""
        urls = [
            "https://github.com/user/repo1",
            "https://github.com/user/repo1/",      # same repo
        ] + self.BASE[1:]
        r = validate_url_list(urls)
        assert r["count"] == 5
        assert len(r["duplicates"]) == 1

    def test_all_duplicates_returns_count_one(self):
        """If the user enters the same URL 5 times, count should be 1."""
        urls = ["https://github.com/user/myrepo"] * 5
        r = validate_url_list(urls)
        assert r["count"] == 1
        assert len(r["duplicates"]) == 4

    def test_response_has_warnings_key(self):
        """validate_url_list must always return a 'warnings' key."""
        r = validate_url_list(self.BASE)
        assert "warnings" in r
        assert isinstance(r["warnings"], list)

    # ── Other existing tests ──────────────────────────────────

    def test_too_many_trimmed(self):
        urls = [f"https://github.com/user/repo{i}" for i in range(1, 12)]
        r = validate_url_list(urls)
        assert r["count"] == 10

    def test_invalid_url_reported(self):
        urls = self.BASE + ["not-a-url"]
        r = validate_url_list(urls)
        assert any("not-a-url" in e["url"] for e in r["errors"])
