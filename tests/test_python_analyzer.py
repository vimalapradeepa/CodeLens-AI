"""Tests for Python AST analyzer."""
import tempfile, os, pytest
from services.python_analyzer import analyze_python_file


def _write(tmp, fname, src):
    p = os.path.join(tmp, fname)
    with open(p, "w") as f: f.write(src)
    return p


class TestPythonAnalyzer:

    def test_simple_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "class Foo:\n    def bar(self): pass\n    def baz(self): pass\n"
            p = _write(tmp, "f.py", src)
            r = analyze_python_file(p, "f.py")
            assert r["error"] is None
            assert len(r["classes"]) == 1
            cls = r["classes"][0]
            assert cls["name"] == "Foo"
            assert "bar" in cls["methods"]
            assert "baz" in cls["methods"]

    def test_standalone_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "def hello(): pass\ndef world(): pass\n"
            p = _write(tmp, "f.py", src)
            r = analyze_python_file(p, "f.py")
            assert "hello" in r["functions"]
            assert "world" in r["functions"]

    def test_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "import os\nimport sys\nfrom pathlib import Path\n"
            p = _write(tmp, "f.py", src)
            r = analyze_python_file(p, "f.py")
            assert "os" in r["imports"]
            assert "sys" in r["imports"]

    def test_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "class Base: pass\nclass Child(Base): pass\n"
            p = _write(tmp, "f.py", src)
            r = analyze_python_file(p, "f.py")
            names = [c["name"] for c in r["classes"]]
            assert "Base" in names
            child = next(c for c in r["classes"] if c["name"] == "Child")
            assert "Base" in child["bases"]

    def test_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "bad.py", "def (: pass\n")
            r = analyze_python_file(p, "bad.py")
            assert r["error"] is not None
