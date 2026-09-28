"""Tests for repository scanner and language detector."""
import os
import tempfile
import pytest
from services.language_detector import detect_language, detect_repo_languages
from services.repository_scanner import scan_repository


def _make_tree(tmp, files: dict):
    """Create a temp dir tree from {rel_path: content}."""
    for rel, content in files.items():
        abs_p = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(abs_p), exist_ok=True)
        with open(abs_p, "w", encoding="utf-8") as f:
            f.write(content)


class TestLanguageDetector:
    def test_python(self):  assert detect_language("foo.py")   == "Python"
    def test_java(self):    assert detect_language("Bar.java") == "Java"
    def test_unknown(self): assert detect_language("data.csv") == "Unknown"
    def test_js(self):      assert detect_language("app.js")   == "JavaScript"

    def test_walk_ignores_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(tmp, {
                "src/main.py": "x=1",
                ".git/config": "[core]",
            })
            r = detect_repo_languages(tmp)
            paths = [t[1] for t in r["file_list"]]
            assert not any(".git" in p for p in paths)

    def test_lang_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(tmp, {
                "a.py": "x=1", "b.py": "y=2", "c.java": "class C{}",
            })
            r = detect_repo_languages(tmp)
            assert r["lang_counts"].get("Python") == 2
            assert r["lang_counts"].get("Java")   == 1


class TestRepositoryScanner:
    def test_scan_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_tree(tmp, {
                "main.py":    "class App:\n    def run(self): pass\n",
                "Helper.java":"class Helper{}\n",
                "README.md":  "# hi",
            })
            result = scan_repository(tmp, "test-repo")
            assert result["total_files"] == 3
            assert result["source_files"] == 2
            py_files = [f for f in result["files"] if f["language"] == "Python"]
            assert len(py_files) == 1

    def test_empty_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = scan_repository(tmp, "empty")
            assert result["total_files"] == 0
            assert result["source_files"] == 0
