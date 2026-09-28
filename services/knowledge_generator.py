"""
Knowledge Generator (Agent 1 component)
Assembles the structured JSON knowledge document for a repository.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def generate_knowledge(metadata: Dict, scan_result: Dict, statistics: Dict) -> Dict:
    """
    Merge metadata + scan + statistics into a single structured knowledge dict.
    """
    repo_name = metadata.get("full_name", metadata.get("name", "unknown"))

    knowledge = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repository": {
            "name": metadata.get("name", ""),
            "full_name": repo_name,
            "owner": metadata.get("owner", ""),
            "url": metadata.get("url", ""),
            "description": metadata.get("description", ""),
            "topics": metadata.get("topics", []),
            "primary_language": metadata.get("primary_language", "Unknown"),
            "languages": metadata.get("languages", {}),
            "stars": metadata.get("stars", 0),
            "forks": metadata.get("forks", 0),
            "size_kb": metadata.get("size_kb", 0),
            "default_branch": metadata.get("default_branch", "main"),
            "created_at": metadata.get("created_at", ""),
            "updated_at": metadata.get("updated_at", ""),
        },
        "statistics": statistics,
        "files": scan_result.get("files", []),
        "scan_errors": scan_result.get("scan_errors", []),
        "unsupported_files_count": len(scan_result.get("unsupported", [])),
    }
    return knowledge


def save_knowledge(knowledge: Dict, owner: str, name: str) -> str:
    """
    Save knowledge dict as JSON to output directory.
    Returns the path to the saved file.
    """
    repo_dir = os.path.join(OUTPUT_DIR, f"{owner}__{name}")
    os.makedirs(repo_dir, exist_ok=True)
    out_path = os.path.join(repo_dir, "knowledge.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(knowledge, fh, indent=2, default=str)
    logger.info("Knowledge saved → %s", out_path)
    return out_path


def load_knowledge(owner: str, name: str) -> Dict:
    """Load previously saved knowledge JSON."""
    path = os.path.join(OUTPUT_DIR, f"{owner}__{name}", "knowledge.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
