"""
Keyword Filter & Relevance Scoring Service

Implements a transparent, weight-based relevance algorithm.
Weights are configurable via config.py SCORING_WEIGHTS.
"""

import re
import math
import logging
from typing import List, Dict

from config import SCORING_WEIGHTS

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric chars."""
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]


def _keyword_tokens(keywords_str: str) -> List[str]:
    """Parse comma/space separated keywords into a token list."""
    raw = re.split(r'[,\s]+', keywords_str.strip())
    return [k.lower().strip() for k in raw if k.strip()]


def _match_score(tokens: List[str], kw_tokens: List[str]) -> float:
    """
    TF-inspired score: fraction of keyword tokens found in text tokens.
    Returns 0.0–1.0.
    """
    if not kw_tokens or not tokens:
        return 0.0
    token_set = set(tokens)
    matches = sum(1 for k in kw_tokens if k in token_set)
    return matches / len(kw_tokens)


def score_repository(metadata: Dict, keywords: str) -> float:
    """
    Compute a 0–100 relevance score for a repository against the keywords.

    Scoring breakdown (configurable weights in config.py):
      name        : repo name token match
      description : description token match
      topics      : topics match
      readme      : README excerpt match
      code        : extracted code entities match (if available)
      language    : primary language match
    """
    if not keywords.strip():
        return 50.0   # neutral when no keywords given

    kw_tokens = _keyword_tokens(keywords)
    w = SCORING_WEIGHTS

    # --- name ---
    name_score = _match_score(_tokenise(metadata.get("name", "")), kw_tokens)

    # --- description ---
    desc_score = _match_score(_tokenise(metadata.get("description", "")), kw_tokens)

    # --- topics ---
    topics_text = " ".join(metadata.get("topics", []))
    topics_score = _match_score(_tokenise(topics_text), kw_tokens)

    # --- readme ---
    readme_score = _match_score(_tokenise(metadata.get("readme", "")), kw_tokens)

    # --- code entities (injected by Agent 1 into metadata after analysis) ---
    code_text = metadata.get("code_entities_text", "")
    code_score = _match_score(_tokenise(code_text), kw_tokens)

    # --- language ---
    lang_text = metadata.get("primary_language") or ""
    lang_score = _match_score(_tokenise(lang_text), kw_tokens)

    raw = (
        w["name"]        * name_score
        + w["description"] * desc_score
        + w["topics"]      * topics_score
        + w["readme"]      * readme_score
        + w["code"]        * code_score
        + w["language"]    * lang_score
    )

    # Boost: partial keyword matches found across multiple fields get a small bonus
    fields_with_match = sum([
        name_score > 0,
        desc_score > 0,
        topics_score > 0,
        readme_score > 0,
        code_score > 0,
        lang_score > 0,
    ])
    breadth_bonus = min(fields_with_match * 0.02, 0.10)   # up to +10%

    final = min(1.0, raw + breadth_bonus)
    return round(final * 100, 2)


def filter_and_score(repos: List[Dict], keywords: str) -> List[Dict]:
    """
    Score all repos and return those with score > 0, sorted descending.
    Repositories with errors are excluded from ranking but kept in result
    with score=0 for UI transparency.
    """
    scored = []
    for repo in repos:
        if repo.get("error"):
            repo["relevance_score"] = 0.0
            repo["matched"] = False
        else:
            score = score_repository(repo, keywords)
            repo["relevance_score"] = score
            repo["matched"] = score > 0
        scored.append(repo)

    scored.sort(key=lambda r: r["relevance_score"], reverse=True)
    return scored
