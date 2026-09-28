"""Tests for Java analyzer (javalang + regex fallback)."""
import tempfile, os
from services.java_analyzer import analyze_java_file


def _write(tmp, src):
    p = os.path.join(tmp, "Test.java")
    with open(p, "w") as f: f.write(src)
    return p


class TestJavaAnalyzer:

    def test_simple_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "package com.example;\npublic class Foo { public void bar() {} }\n"
            p = _write(tmp, src)
            r = analyze_java_file(p, "Test.java")
            assert len(r["classes"]) >= 1
            assert r["classes"][0]["name"] == "Foo"

    def test_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "import java.util.List;\nimport java.io.File;\nclass X{}\n"
            p = _write(tmp, src)
            r = analyze_java_file(p, "Test.java")
            assert any("java.util.List" in i for i in r["imports"])

    def test_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "interface Readable { String read(); }\n"
            p = _write(tmp, src)
            r = analyze_java_file(p, "Test.java")
            assert any(i["name"] == "Readable" for i in r["interfaces"])

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "")
            r = analyze_java_file(p, "empty.java")
            # Should not crash
            assert isinstance(r, dict)
