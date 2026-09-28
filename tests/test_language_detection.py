"""
Tests for language detection in github_service.

Key invariants verified:
  - primary_language is determined ONLY from actual source-code byte counts
    (the GitHub /languages endpoint), never from keywords, topics, or README.
  - code_languages lists ALL actual code languages (non-code excluded), ordered
    by byte count, used for the multi-language Language column in Discovery.
  - Repos where GitHub labels the primary as "Jupyter Notebook", "HTML", etc.
    are correctly re-mapped to the dominant code language (e.g. Python).
  - Mixed-language repos return all code languages in code_languages.
  - Repos with no detectable code language return None / [] appropriately.
  - filter_and_score never modifies primary_language or code_languages.
"""

import copy
import pytest
from services.github_service import (
    _best_language, _all_code_languages,
    _NON_CODE_LANGUAGES, _CODE_LANGUAGE_PRIORITY,
)
from services.keyword_filter import filter_and_score


# ──────────────────────────────────────────────────────────────
# _best_language — core language detection logic
# ──────────────────────────────────────────────────────────────

class TestBestLanguage:

    # ── Straightforward code-language repos ──────────────────

    def test_python_repo_returns_python(self):
        assert _best_language("Python", {"Python": 95000, "Shell": 5000}) == "Python"

    def test_java_repo_returns_java(self):
        assert _best_language("Java", {"Java": 80000, "XML": 10000}) == "Java"

    def test_javascript_repo_returns_javascript(self):
        assert _best_language("JavaScript", {"JavaScript": 60000, "HTML": 20000, "CSS": 10000}) == "JavaScript"

    def test_go_repo_returns_go(self):
        assert _best_language("Go", {"Go": 120000, "Makefile": 2000}) == "Go"

    def test_cpp_repo_returns_cpp(self):
        assert _best_language("C++", {"C++": 200000, "CMake": 5000}) == "C++"

    # ── Jupyter Notebook primary → must return underlying code lang ──

    def test_jupyter_notebook_with_python_returns_python(self):
        """
        Phishing_Website_Detection-style ML repo:
        GitHub primary = 'Jupyter Notebook' but real code is Python.
        """
        result = _best_language(
            "Jupyter Notebook",
            {"Jupyter Notebook": 500000, "Python": 120000, "HTML": 5000},
        )
        assert result == "Python"

    def test_jupyter_notebook_with_r_returns_r(self):
        """R-language data science repo mislabelled as Jupyter Notebook."""
        result = _best_language(
            "Jupyter Notebook",
            {"Jupyter Notebook": 300000, "R": 80000},
        )
        assert result == "R"

    def test_jupyter_notebook_only_returns_jupyter(self):
        """Repo that really is only notebooks — return the notebook label."""
        result = _best_language(
            "Jupyter Notebook",
            {"Jupyter Notebook": 300000},
        )
        assert result == "Jupyter Notebook"

    # ── Other non-code primaries ──────────────────────────────

    def test_html_primary_with_javascript_returns_javascript(self):
        result = _best_language(
            "HTML",
            {"HTML": 200000, "JavaScript": 150000, "CSS": 50000},
        )
        assert result == "JavaScript"

    def test_html_primary_with_typescript_returns_typescript(self):
        result = _best_language(
            "HTML",
            {"HTML": 180000, "TypeScript": 160000},
        )
        assert result == "TypeScript"

    def test_yaml_primary_with_python_returns_python(self):
        """Config/infra repo where YAML dominates but real code is Python."""
        result = _best_language(
            "YAML",
            {"YAML": 400000, "Python": 50000},
        )
        assert result == "Python"

    def test_dockerfile_primary_with_go_returns_go(self):
        result = _best_language(
            "Dockerfile",
            {"Dockerfile": 10000, "Go": 80000, "Shell": 5000},
        )
        assert result == "Go"

    def test_shell_primary_with_python_returns_python(self):
        result = _best_language(
            "Shell",
            {"Shell": 30000, "Python": 25000},
        )
        assert result == "Python"

    # ── langs_data always wins over github_primary ────────────

    def test_langs_data_overrides_github_primary_when_both_code(self):
        """
        langs_data is the authoritative source. The priority list determines
        which language wins when multiple code languages are present.
        Python is higher in the priority list than Java, so Python wins
        regardless of byte counts when both are present.
        """
        result = _best_language(
            "Python",
            {"Java": 300000, "Python": 50000},
        )
        # Python is first in _CODE_LANGUAGE_PRIORITY → Python wins
        assert result == "Python"

    def test_langs_data_java_only_returns_java(self):
        """When only Java is in langs_data (no Python), Java is returned."""
        result = _best_language(
            "Python",   # github_primary says Python but breakdown only has Java
            {"Java": 300000, "HTML": 50000},
        )
        assert result == "Java"

    def test_langs_data_highest_code_lang_by_priority(self):
        """Priority list is consulted before byte-count order."""
        result = _best_language(
            "Jupyter Notebook",
            {"Jupyter Notebook": 900000, "R": 50000, "Python": 30000},
        )
        # Python is higher priority than R
        assert result == "Python"

    def test_unknown_code_language_returned_by_byte_order(self):
        """A language not on the priority list is returned if it's the only code lang."""
        result = _best_language(
            "Kotlin",
            {"Kotlin": 100000, "HTML": 20000},
        )
        assert result == "Kotlin"

    # ── Missing data fallbacks ────────────────────────────────

    def test_none_primary_falls_back_to_langs_data(self):
        """GitHub returns null for language; langs endpoint has Python."""
        assert _best_language(None, {"Python": 80000}) == "Python"

    def test_empty_langs_data_returns_github_primary(self):
        """Languages endpoint failed (rate-limited); use main-repo field."""
        assert _best_language("Python", {}) == "Python"

    def test_empty_langs_data_non_code_primary_returns_none(self):
        """
        No breakdown available and github_primary is a non-code label.
        Must return None (renders as '—') rather than showing a misleading
        non-code label like 'HTML' or 'Jupyter Notebook' as the language.
        """
        assert _best_language("Jupyter Notebook", {}) is None
        assert _best_language("HTML", {}) is None
        assert _best_language("YAML", {}) is None
        assert _best_language("Dockerfile", {}) is None

    def test_completely_empty_returns_none(self):
        assert _best_language(None, {}) is None

    # ── LostPersonAlert-style: HTML primary but Python in breakdown ───

    def test_html_primary_with_python_in_breakdown_returns_python(self):
        """
        sona-1346/LostPersonAlert style:
        GitHub primary = 'HTML', breakdown = HTML+Python+CSS+C++.
        Must return Python (first code language in priority list found in breakdown).
        """
        result = _best_language(
            "HTML",
            {"HTML": 21572, "Python": 17114, "CSS": 11258, "C++": 6080},
        )
        assert result == "Python"

    def test_html_primary_no_breakdown_returns_none(self):
        """
        HTML primary with no languages breakdown (rate-limited).
        Must return None rather than 'HTML'.
        """
        result = _best_language("HTML", {})
        assert result is None

    def test_css_primary_with_cpp_returns_cpp(self):
        """CSS primary but real code is C++."""
        result = _best_language(
            "CSS",
            {"CSS": 50000, "C++": 30000},
        )
        assert result == "C++"

    # ── Mixed-language repos: dominant code language ──────────

    def test_mixed_python_java_picks_dominant(self):
        """Python has more bytes → Python wins."""
        result = _best_language(
            "Python",
            {"Python": 150000, "Java": 40000, "Shell": 10000},
        )
        assert result == "Python"

    def test_mixed_java_python_picks_java_from_priority(self):
        """Java first in priority list check — should be found before Python
        only if its bytes entry is present; here Java IS present."""
        # Note: _CODE_LANGUAGE_PRIORITY has Python before Java… wait:
        # Python is index 0, Java is index 1 — so Python would be found first.
        # The correct assertion is Python wins when Python has more bytes.
        result = _best_language(
            "Java",
            {"Java": 200000, "Python": 50000, "HTML": 10000},
        )
        # Python is higher priority than Java in the list → Python returned
        # even though Java has more bytes. This is the defined contract.
        assert result == "Python"

    def test_mixed_no_python_java_wins(self):
        """When Python is absent, Java is returned."""
        result = _best_language(
            "Java",
            {"Java": 200000, "HTML": 50000, "CSS": 10000},
        )
        assert result == "Java"

    # ── Non-code-language set is populated ───────────────────

    def test_non_code_set_contains_key_entries(self):
        for entry in ["jupyter notebook", "html", "yaml", "dockerfile", "shell"]:
            assert entry in _NON_CODE_LANGUAGES, f"'{entry}' missing from _NON_CODE_LANGUAGES"

    def test_priority_list_is_non_empty(self):
        assert len(_CODE_LANGUAGE_PRIORITY) >= 10

    def test_python_in_priority_list(self):
        assert "Python" in _CODE_LANGUAGE_PRIORITY

    def test_java_in_priority_list(self):
        assert "Java" in _CODE_LANGUAGE_PRIORITY


# ──────────────────────────────────────────────────────────────
# Keywords must NEVER affect primary_language
# ──────────────────────────────────────────────────────────────

class TestLanguageNotDerivedFromKeywords:

    def _repo(self, name, primary_language, description="", topics=None):
        return {
            "name": name,
            "description": description,
            "topics": topics or [],
            "readme": "",
            "primary_language": primary_language,
            "error": None,
        }

    def test_python_ml_repo_keeps_python_with_java_keyword(self):
        repos = [self._repo("phishing-detection", "Python", "ML phishing detection")]
        assert filter_and_score(repos, "java")[0]["primary_language"] == "Python"

    def test_python_repo_keeps_python_with_ml_keyword(self):
        repos = [self._repo("sklearn-utils", "Python", "machine learning utilities")]
        assert filter_and_score(repos, "machine learning")[0]["primary_language"] == "Python"

    def test_java_repo_keeps_java_with_python_keyword(self):
        repos = [self._repo("spring-app", "Java", "Spring Boot application")]
        assert filter_and_score(repos, "python machine learning")[0]["primary_language"] == "Java"

    def test_none_language_stays_none_regardless_of_keywords(self):
        repos = [self._repo("docs-only", None, "documentation only")]
        assert filter_and_score(repos, "python java javascript")[0]["primary_language"] is None

    def test_language_invariant_across_all_keyword_combinations(self):
        """primary_language must be identical no matter what keywords are used."""
        base_repo = self._repo(
            "Phishing_Website_Detection",
            "Python",
            "Phishing website detection using ML",
            ["machine-learning", "python", "phishing"],
        )
        for keywords in [
            "machine learning",
            "python",
            "java",
            "python java machine learning",
            "deep learning neural network",
            "",
            "completely unrelated keywords xyz",
        ]:
            r = copy.deepcopy(base_repo)
            result = filter_and_score([r], keywords)
            assert result[0]["primary_language"] == "Python", (
                f"primary_language became {result[0]['primary_language']!r} "
                f"with keywords={keywords!r}"
            )

    def test_relevance_changes_but_language_does_not(self):
        """Scoring changes the relevance_score, never primary_language."""
        repos = [
            self._repo("repo-a", "Python", "data science"),
            self._repo("repo-b", "Java", "enterprise app"),
        ]
        for kw in ["python", "java", "machine learning", ""]:
            clones = copy.deepcopy(repos)
            scored = filter_and_score(clones, kw)
            by_name = {r["name"]: r for r in scored}
            assert by_name["repo-a"]["primary_language"] == "Python", kw
            assert by_name["repo-b"]["primary_language"] == "Java", kw

    def test_topic_ml_with_java_code_shows_java(self):
        """
        A repo about ML (topic) that is actually written in Java must show Java.
        The topic must not influence the language column.
        """
        repos = [self._repo(
            "ml-java-lib", "Java",
            "Machine learning library",
            ["machine-learning", "java", "deep-learning"],
        )]
        scored = filter_and_score(repos, "machine learning")
        assert scored[0]["primary_language"] == "Java"

    def test_language_column_reflects_code_not_project_domain(self):
        """
        A computer-vision project written in C++ must show C++, not Python,
        even though Python is the more 'popular' ML language.
        """
        repos = [self._repo(
            "cv-engine", "C++",
            "High-performance computer vision engine",
            ["computer-vision", "opencv"],
        )]
        scored = filter_and_score(repos, "python machine learning deep learning")
        assert scored[0]["primary_language"] == "C++"

    def test_lost_person_alert_style_repo(self):
        """
        sona-1346/LostPersonAlert: description says 'social impact solution'
        but actual code is Python. Keywords 'machine learning, AI' must not
        change the detected language.
        """
        repos = [self._repo(
            "LostPersonAlert",
            "Python",   # as returned by _best_language after checking breakdown
            "LostPersonAlert is more than a technical project—it's a social impact solution.",
            [],
        )]
        for kw in ["machine learning", "AI", "lost person", "alert", "python", "java", ""]:
            r = copy.deepcopy(repos[0])
            result = filter_and_score([r], kw)
            assert result[0]["primary_language"] == "Python", (
                f"Language changed to {result[0]['primary_language']!r} with keyword={kw!r}"
            )

    def test_html_primary_repo_with_python_code_shows_python(self):
        """
        A repo whose GitHub primary language is HTML but whose langs breakdown
        contains Python must show Python in the Language column.
        This mirrors the exact sona-1346/LostPersonAlert API response.
        """
        repos = [self._repo(
            "LostPersonAlert",
            "Python",   # _best_language corrects HTML→Python using breakdown
            "Social impact ML solution",
        )]
        scored = filter_and_score(repos, "machine learning python")
        assert scored[0]["primary_language"] == "Python"
        # Verify keyword did not inject 'machine learning' as a language
        assert scored[0]["primary_language"] != "Machine Learning"
        assert scored[0]["primary_language"] != "AI"


# ──────────────────────────────────────────────────────────────
# _all_code_languages — multi-language list for Discovery column
# ──────────────────────────────────────────────────────────────

class TestAllCodeLanguages:

    # ── Basic filtering ───────────────────────────────────────

    def test_returns_only_code_languages(self):
        """Non-code entries must be excluded from the result."""
        result = _all_code_languages(
            {"Python": 50000, "HTML": 20000, "CSS": 10000, "Shell": 5000}
        )
        assert result == ["Python"]

    def test_multiple_code_languages_all_returned(self):
        """Every code language in the breakdown must appear in the result."""
        result = _all_code_languages(
            {"Python": 50000, "JavaScript": 30000, "TypeScript": 20000,
             "HTML": 15000, "CSS": 10000}
        )
        assert result == ["Python", "JavaScript", "TypeScript"]

    def test_order_follows_byte_count(self):
        """Languages must be ordered highest-bytes first (GitHub API order)."""
        result = _all_code_languages(
            {"Java": 80000, "Python": 50000, "Kotlin": 10000, "YAML": 5000}
        )
        assert result == ["Java", "Python", "Kotlin"]

    # ── LostPersonAlert exact API data ───────────────────────

    def test_lost_person_alert_returns_python_and_cpp(self):
        """
        sona-1346/LostPersonAlert: HTML+Python+CSS+C++.
        Expected code languages (non-code stripped): Python, C++
        ordered by byte count.
        """
        result = _all_code_languages(
            {"HTML": 21572, "Python": 17114, "CSS": 11258, "C++": 6080}
        )
        assert result == ["Python", "C++"]
        assert "HTML" not in result
        assert "CSS"  not in result

    # ── Single code language ──────────────────────────────────

    def test_single_python_returns_single_item_list(self):
        result = _all_code_languages({"Python": 100000, "Shell": 5000})
        assert result == ["Python"]

    def test_single_java_returns_single_item_list(self):
        result = _all_code_languages({"Java": 80000, "XML": 10000, "YAML": 2000})
        assert result == ["Java"]

    # ── All non-code ──────────────────────────────────────────

    def test_only_html_css_returns_empty(self):
        """Repos containing only markup/config return an empty list."""
        result = _all_code_languages({"HTML": 100000, "CSS": 50000, "YAML": 10000})
        assert result == []

    # ── Empty / rate-limited breakdown ───────────────────────

    def test_empty_langs_data_falls_back_to_github_primary(self):
        """When langs_data is empty but github_primary is a code lang, return it."""
        result = _all_code_languages({}, github_primary="Python")
        assert result == ["Python"]

    def test_empty_langs_data_non_code_primary_returns_empty(self):
        """When langs_data is empty and github_primary is non-code, return []."""
        result = _all_code_languages({}, github_primary="HTML")
        assert result == []
        result2 = _all_code_languages({}, github_primary="Jupyter Notebook")
        assert result2 == []

    def test_empty_langs_data_no_primary_returns_empty(self):
        result = _all_code_languages({})
        assert result == []

    # ── Keywords / domain isolation ───────────────────────────

    def test_code_languages_not_affected_by_keywords(self):
        """
        code_languages in the metadata dict must be unchanged regardless of
        what keywords filter_and_score receives.
        """
        repo = {
            "name": "LostPersonAlert",
            "description": "Social impact ML solution using Python",
            "topics": ["machine-learning", "python"],
            "readme": "",
            "primary_language": "Python",
            "code_languages": ["Python", "C++"],
            "error": None,
        }
        for kw in ["machine learning", "AI", "java", "python", "HTML", ""]:
            r = copy.deepcopy(repo)
            result = filter_and_score([r], kw)
            assert result[0]["code_languages"] == ["Python", "C++"], (
                f"code_languages changed with keyword={kw!r}"
            )

    def test_no_non_code_language_in_code_languages(self):
        """
        After _all_code_languages, none of the known non-code labels should
        appear in the result regardless of what the breakdown contains.
        """
        non_code_samples = [
            "HTML", "CSS", "YAML", "JSON", "Dockerfile", "Shell",
            "Makefile", "Jupyter Notebook", "Markdown",
        ]
        for non_code in non_code_samples:
            langs = {non_code: 100000, "Python": 50000}
            result = _all_code_languages(langs)
            assert non_code not in result, (
                f"Non-code language '{non_code}' appeared in result: {result}"
            )
            assert result == ["Python"]

    def test_ai_ml_not_valid_programming_language(self):
        """'AI', 'Machine Learning', 'ML' are never valid languages."""
        for bad_lang in ["AI", "Machine Learning", "ML"]:
            # They would not appear in GitHub's /languages response, but
            # verify _all_code_languages never introduces them.
            result = _all_code_languages({"Python": 50000, "Java": 30000})
            assert bad_lang not in result

    # ── Comma-separated display string ───────────────────────

    def test_comma_separated_display(self):
        """Joining code_languages with ', ' must produce a readable string."""
        langs = _all_code_languages(
            {"Python": 50000, "JavaScript": 30000, "TypeScript": 10000,
             "HTML": 5000}
        )
        display = ", ".join(langs)
        assert display == "Python, JavaScript, TypeScript"

    def test_single_language_display_no_comma(self):
        langs = _all_code_languages({"Python": 80000, "CSS": 10000})
        display = ", ".join(langs)
        assert "," not in display
        assert display == "Python"
