"""
Python Analyzer (Agent 1 component)
Uses the built-in ast module to extract code entities from Python source files.
No external parser required.
"""

import ast
import os
import logging
from typing import Dict, List, Optional

from config import MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Visitor
# ──────────────────────────────────────────────

class _PythonEntityVisitor(ast.NodeVisitor):
    """AST visitor that extracts classes, methods, functions, imports, calls."""

    def __init__(self):
        self.classes: List[Dict] = []
        self.functions: List[str] = []
        self.imports: List[str] = []
        self.calls: List[str] = []
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = [_name_of(b) for b in node.bases]
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        self.classes.append({
            "name": node.name,
            "methods": methods,
            "bases": bases,
            "line": node.lineno,
        })
        old = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._current_class is None:
            self.functions.append(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}" if module else alias.name)

    def visit_Call(self, node: ast.Call):
        name = _name_of(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)


def _name_of(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    return ""


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def analyze_python_file(abs_path: str, rel_path: str) -> Dict:
    """
    Parse a single Python file and return extracted entities.
    Returns a dict with 'error' key on failure.
    """
    result = {
        "path": rel_path,
        "language": "Python",
        "classes": [],
        "functions": [],
        "imports": [],
        "calls": [],
        "error": None,
    }

    try:
        size = os.path.getsize(abs_path)
        if size > MAX_FILE_SIZE_BYTES:
            result["error"] = f"File too large ({size} bytes), skipped."
            return result

        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()

        tree = ast.parse(source, filename=rel_path)
        visitor = _PythonEntityVisitor()
        visitor.visit(tree)

        result["classes"]   = visitor.classes
        result["functions"] = visitor.functions
        result["imports"]   = list(set(visitor.imports))
        result["calls"]     = list(set(visitor.calls))

    except SyntaxError as exc:
        result["error"] = f"SyntaxError: {exc}"
        logger.debug("Syntax error in %s: %s", rel_path, exc)
    except Exception as exc:
        result["error"] = f"Parse error: {exc}"
        logger.warning("Failed to parse %s: %s", rel_path, exc)

    return result


def analyze_python_files(file_list: List[tuple]) -> List[Dict]:
    """
    Analyze a list of (abs_path, rel_path) Python tuples.
    Returns list of file analysis dicts.
    """
    results = []
    for abs_p, rel_p in file_list:
        results.append(analyze_python_file(abs_p, rel_p))
    return results
