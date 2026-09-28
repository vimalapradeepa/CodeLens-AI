"""
Java Analyzer (Agent 1 component)
Uses the `javalang` library to parse Java source files.
Falls back gracefully to regex-based extraction if javalang is unavailable or fails.
"""

import os
import re
import logging
from typing import Dict, List, Optional

from config import MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)

try:
    import javalang
    _JAVALANG_AVAILABLE = True
except ImportError:
    _JAVALANG_AVAILABLE = False
    logger.warning("javalang not installed — Java analysis will use regex fallback.")


# ──────────────────────────────────────────────
# javalang-based extraction
# ──────────────────────────────────────────────

def _analyze_with_javalang(source: str) -> Dict:
    tree = javalang.parse.parse(source)

    package = ""
    if tree.package:
        package = tree.package.name

    imports = [imp.path for imp in (tree.imports or [])]

    classes = []
    interfaces = []

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            methods = [m.name for m in node.methods] if node.methods else []
            fields  = [f.declarators[0].name for f in (node.fields or []) if f.declarators]
            extends = node.extends.name if node.extends else None
            implements = [i.name for i in (node.implements or [])]
            classes.append({
                "name": node.name,
                "methods": methods,
                "fields": fields,
                "extends": extends,
                "implements": implements,
            })
        elif isinstance(node, javalang.tree.InterfaceDeclaration):
            methods = [m.name for m in (node.methods or [])]
            interfaces.append({"name": node.name, "methods": methods})

    return {
        "package": package,
        "imports": imports,
        "classes": classes,
        "interfaces": interfaces,
        "error": None,
    }


# ──────────────────────────────────────────────
# Regex fallback
# ──────────────────────────────────────────────

def _analyze_with_regex(source: str) -> Dict:
    """Best-effort extraction when javalang is unavailable."""
    package_m = re.search(r'^\s*package\s+([\w.]+)\s*;', source, re.MULTILINE)
    package = package_m.group(1) if package_m else ""

    imports = re.findall(r'^\s*import\s+([\w.*]+)\s*;', source, re.MULTILINE)

    class_pattern = re.compile(
        r'(?:public|private|protected|abstract|final|static)?\s*'
        r'class\s+(\w+)'
        r'(?:\s+extends\s+(\w+))?'
        r'(?:\s+implements\s+([\w,\s]+))?'
    )
    classes = []
    for m in class_pattern.finditer(source):
        classes.append({
            "name": m.group(1),
            "extends": m.group(2),
            "implements": [i.strip() for i in (m.group(3) or "").split(",") if i.strip()],
            "methods": [],
            "fields": [],
        })

    iface_pattern = re.compile(r'interface\s+(\w+)')
    interfaces = [{"name": m.group(1), "methods": []} for m in iface_pattern.finditer(source)]

    method_pattern = re.compile(
        r'(?:public|private|protected|static|final|abstract|synchronized)[\w\s<>\[\]]*'
        r'\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
    )
    all_methods = [m.group(1) for m in method_pattern.finditer(source)]
    if classes:
        classes[0]["methods"] = all_methods   # rough assignment to first class

    return {
        "package": package,
        "imports": imports,
        "classes": classes,
        "interfaces": interfaces,
        "error": None,
    }


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def analyze_java_file(abs_path: str, rel_path: str) -> Dict:
    result = {
        "path": rel_path,
        "language": "Java",
        "package": "",
        "classes": [],
        "interfaces": [],
        "imports": [],
        "error": None,
    }

    try:
        size = os.path.getsize(abs_path)
        if size > MAX_FILE_SIZE_BYTES:
            result["error"] = f"File too large ({size} bytes), skipped."
            return result

        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()

        if _JAVALANG_AVAILABLE:
            try:
                parsed = _analyze_with_javalang(source)
            except Exception as exc:
                logger.debug("javalang failed on %s: %s — using regex fallback", rel_path, exc)
                parsed = _analyze_with_regex(source)
                parsed["error"] = f"javalang parse error (used regex fallback): {exc}"
        else:
            parsed = _analyze_with_regex(source)

        result.update(parsed)

    except Exception as exc:
        result["error"] = f"File read error: {exc}"
        logger.warning("Failed to read %s: %s", rel_path, exc)

    return result


def analyze_java_files(file_list: List[tuple]) -> List[Dict]:
    results = []
    for abs_p, rel_p in file_list:
        results.append(analyze_java_file(abs_p, rel_p))
    return results
