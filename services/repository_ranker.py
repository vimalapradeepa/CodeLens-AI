"""
Repository Ranker
Produces the Top-N ranked list from scored repositories.
"""

from typing import List, Dict


def rank_repositories(scored_repos: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    Accept a list of scored repos (from keyword_filter.filter_and_score),
    assign rank positions, and return the top_n.

    Each repo gets:
        rank           : 1-based position
        relevance_pct  : human-readable percentage string
    """
    # Sort again in case caller skipped keyword_filter sort
    ranked = sorted(scored_repos, key=lambda r: r.get("relevance_score", 0), reverse=True)
    ranked = ranked[:top_n]

    for i, repo in enumerate(ranked, start=1):
        repo["rank"] = i
        score = repo.get("relevance_score", 0)
        repo["relevance_pct"] = f"{score:.1f}%"

    return ranked


def get_ranking_summary(ranked_repos: List[Dict], keywords: str, total_submitted: int) -> Dict:
    """Build a summary dict for the discovery page."""
    matching = sum(1 for r in ranked_repos if r.get("matched", False))
    return {
        "keywords": keywords,
        "total_submitted": total_submitted,
        "total_matching": matching,
        "total_ranked": len(ranked_repos),
        "top_repos": ranked_repos,
    }
