"""
Language Detector
Walks a cloned repository and detects programming languages by file extension.
Respects the IGNORE_DIRS list from config.
"""

import os
from typing import Dict, List, Tuple
from config import IGNORE_DIRS

_EXTENSION_MAP = {
    ".py":   "Python",
    ".java": "Java",
    ".js":   "JavaScript",
    ".ts":   "TypeScript",
    ".cpp":  "C++",
    ".c":    "C",
    ".cs":   "C#",
    ".rb":   "Ruby",
    ".go":   "Go",
    ".rs":   "Rust",
    ".kt":   "Kotlin",
    ".swift":"Swift",
    ".php":  "PHP",
    ".scala":"Scala",
    ".r":    "R",
    ".m":    "MATLAB",
    ".sh":   "Shell",
    ".html": "HTML",
    ".css":  "CSS",
    ".json": "JSON",
    ".xml":  "XML",
    ".yaml": "YAML",
    ".yml":  "YAML",
    ".md":   "Markdown",
}


def _should_ignore(name: str) -> bool:
    return name in IGNORE_DIRS or name.startswith(".")


def walk_source_files(local_path: str) -> List[Tuple[str, str]]:
    """
    Recursively yield (abs_path, relative_path) for every non-ignored file.
    """
    results = []
    for dirpath, dirnames, filenames in os.walk(local_path):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if not _should_ignore(d)]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, local_path)
            results.append((abs_path, rel_path))
    return results


def detect_language(file_path: str) -> str:
    """Return language string for a file path, or 'Unknown'."""
    _, ext = os.path.splitext(file_path.lower())
    return _EXTENSION_MAP.get(ext, "Unknown")


def detect_repo_languages(local_path: str) -> Dict:
    """
    Walk repository and return language breakdown.

    Returns:
        {
            "file_list"   : [(abs_path, rel_path, language), …],
            "lang_counts" : {"Python": 23, "Java": 4, …},
            "total_files" : int,
        }
    """
    files = walk_source_files(local_path)
    lang_counts: Dict[str, int] = {}
    file_list = []

    for abs_p, rel_p in files:
        lang = detect_language(abs_p)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        file_list.append((abs_p, rel_p, lang))

    return {
        "file_list": file_list,
        "lang_counts": lang_counts,
        "total_files": len(file_list),
    }
