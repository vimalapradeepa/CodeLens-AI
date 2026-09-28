"""
Repository Scanner (Agent 1 component)
Orchestrates language detection and per-file analysis for a cloned repository.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from services.language_detector import detect_repo_languages
from services.python_analyzer import analyze_python_file
from services.java_analyzer import analyze_java_file
from config import SUPPORTED_LANGUAGES, MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)

# Number of worker processes for parallel file analysis.
# Use CPU count but cap to avoid overwhelming the system.
_MAX_WORKERS = min(os.cpu_count() or 4, 8)


def _analyze_python_file_worker(args: tuple) -> Dict:
    """Worker function for parallel Python file analysis."""
    abs_path, rel_path = args
    return analyze_python_file(abs_path, rel_path)


def _analyze_java_file_worker(args: tuple) -> Dict:
    """Worker function for parallel Java file analysis."""
    abs_path, rel_path = args
    return analyze_java_file(abs_path, rel_path)


def scan_repository(local_path: str, repo_name: str = "") -> Dict:
    """
    Scan a cloned repository and return a structured result dict.

    Uses parallel processing for file analysis to handle large repositories
    (e.g., apache/airflow with thousands of files) efficiently.

    Returns:
        {
            "repo_name"      : str,
            "local_path"     : str,
            "total_files"    : int,
            "lang_counts"    : {lang: count},
            "source_files"   : int,
            "files"          : [file_result_dict, …],
            "unsupported"    : [rel_path, …],
            "scan_errors"    : [str, …],
        }
    """
    detection = detect_repo_languages(local_path)
    file_list = detection["file_list"]   # (abs_path, rel_path, language)
    lang_counts = detection["lang_counts"]
    total_files = detection["total_files"]

    # Separate files by language for parallel processing
    python_files = [(abs_p, rel_p) for abs_p, rel_p, lang in file_list if lang == "Python"]
    java_files = [(abs_p, rel_p) for abs_p, rel_p, lang in file_list if lang == "Java"]
    unsupported = [rel_p for abs_p, rel_p, lang in file_list if lang not in SUPPORTED_LANGUAGES]

    analyzed_files: List[Dict] = []
    scan_errors: List[str] = []
    source_files = 0

    # Process Python files in parallel
    if python_files:
        logger.info("Analyzing %d Python files in parallel (workers=%d)", len(python_files), _MAX_WORKERS)
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_analyze_python_file_worker, args): args[1] for args in python_files}
            for future in as_completed(futures):
                rel_p = futures[future]
                try:
                    result = future.result()
                    analyzed_files.append(result)
                    source_files += 1
                    if result.get("error"):
                        scan_errors.append(f"{rel_p}: {result['error']}")
                except Exception as exc:
                    logger.warning("Parallel analysis failed for %s: %s", rel_p, exc)
                    analyzed_files.append({
                        "path": rel_p,
                        "language": "Python",
                        "classes": [],
                        "functions": [],
                        "imports": [],
                        "calls": [],
                        "error": f"Parallel analysis error: {exc}",
                    })
                    scan_errors.append(f"{rel_p}: Parallel analysis error: {exc}")

    # Process Java files in parallel
    if java_files:
        logger.info("Analyzing %d Java files in parallel (workers=%d)", len(java_files), _MAX_WORKERS)
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_analyze_java_file_worker, args): args[1] for args in java_files}
            for future in as_completed(futures):
                rel_p = futures[future]
                try:
                    result = future.result()
                    analyzed_files.append(result)
                    source_files += 1
                    if result.get("error"):
                        scan_errors.append(f"{rel_p}: {result['error']}")
                except Exception as exc:
                    logger.warning("Parallel analysis failed for %s: %s", rel_p, exc)
                    analyzed_files.append({
                        "path": rel_p,
                        "language": "Java",
                        "classes": [],
                        "functions": [],
                        "imports": [],
                        "calls": [],
                        "error": f"Parallel analysis error: {exc}",
                    })
                    scan_errors.append(f"{rel_p}: Parallel analysis error: {exc}")

    logger.info(
        "Scan complete for '%s': %d total files, %d source files analysed, %d errors",
        repo_name, total_files, source_files, len(scan_errors),
    )

    return {
        "repo_name": repo_name,
        "local_path": local_path,
        "total_files": total_files,
        "lang_counts": lang_counts,
        "source_files": source_files,
        "files": analyzed_files,
        "unsupported": unsupported,
        "scan_errors": scan_errors,
    }
