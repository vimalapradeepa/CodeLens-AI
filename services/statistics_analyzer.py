"""
Statistics Analyzer (Agent 1 component)
Computes repository-level statistics from scan results.
"""

from typing import Dict, List, Set


def compute_statistics(scan_result: Dict) -> Dict:
    """
    Given the output of repository_scanner.scan_repository,
    compute aggregate statistics.
    """
    files: List[Dict] = scan_result.get("files", [])
    lang_counts = scan_result.get("lang_counts", {})

    total_classes     = 0
    total_methods     = 0
    total_functions   = 0
    total_imports: Set[str] = set()
    total_calls: Set[str] = set()
    dependencies: Set[str] = set()

    python_files = 0
    java_files   = 0

    for f in files:
        lang = f.get("language", "Unknown")
        if lang == "Python":
            python_files += 1
            classes = f.get("classes", [])
            total_classes += len(classes)
            for cls in classes:
                total_methods += len(cls.get("methods", []))
            total_functions += len(f.get("functions", []))
            for imp in f.get("imports", []):
                total_imports.add(imp)
                # Top-level module name as dependency
                dep = imp.split(".")[0]
                if dep:
                    dependencies.add(dep)
            for call in f.get("calls", []):
                total_calls.add(call)

        elif lang == "Java":
            java_files += 1
            for cls in f.get("classes", []):
                total_classes += 1
                total_methods += len(cls.get("methods", []))
            for imp in f.get("imports", []):
                total_imports.add(imp)
                dep = imp.split(".")[0]
                if dep:
                    dependencies.add(dep)

    stats = {
        "total_files":     scan_result.get("total_files", 0),
        "source_files":    scan_result.get("source_files", 0),
        "python_files":    python_files,
        "java_files":      java_files,
        "classes":         total_classes,
        "methods":         total_methods,
        "functions":       total_functions,
        "imports":         len(total_imports),
        "unique_calls":    len(total_calls),
        "dependencies":    sorted(dependencies),
        "lang_counts":     lang_counts,
        "scan_errors":     len(scan_result.get("scan_errors", [])),
    }

    # Code-entity text used later for relevance re-scoring after analysis
    entity_tokens = []
    for f in files:
        for cls in f.get("classes", []):
            if isinstance(cls, dict):
                entity_tokens.append(cls.get("name", ""))
        entity_tokens.extend(f.get("functions", []))
    stats["code_entities_text"] = " ".join(entity_tokens)

    return stats
