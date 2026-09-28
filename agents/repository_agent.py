"""
Agent 1 — Repository Analysis Agent
====================================
Custom agent built from scratch for CodeLens AI.

Responsibility:
  - Accept GitHub URLs + keywords
  - Validate inputs
  - Collect metadata from GitHub API
  - Filter and rank repositories by keyword relevance
  - Clone top-ranked repositories
  - Scan and parse Python/Java source code
  - Compute statistics
  - Generate structured JSON knowledge
  - Generate human-readable report

This agent does NOT depend on CrewAI, AutoGen, or LangGraph.
All analysis logic is implemented in the services/ package.
"""

import logging
import time
from typing import List, Dict, Optional, Callable

from services.url_validator import validate_url_list
from services.github_service import fetch_repos_metadata
from services.keyword_filter import filter_and_score
from services.repository_ranker import rank_repositories, get_ranking_summary
from services.repository_cloner import clone_repository
from services.repository_scanner import scan_repository
from services.statistics_analyzer import compute_statistics
from services.knowledge_generator import generate_knowledge, save_knowledge
from services.report_generator import generate_report, save_report

logger = logging.getLogger(__name__)


class RepositoryAnalysisAgent:
    """
    Agent 1: Repository Analysis Agent

    Usage::

        agent = RepositoryAnalysisAgent()
        result = agent.run(urls=[...], keywords="machine learning, NLP")
    """

    AGENT_NAME = "Repository Analysis Agent"
    AGENT_VERSION = "1.0.0"

    def __init__(self, progress_callback: Optional[Callable[[str, str], None]] = None):
        """
        progress_callback: optional callable(step_name, message) for UI progress updates.
        """
        self.progress_callback = progress_callback or (lambda s, m: None)
        self._state: Dict = {}

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _progress(self, step: str, message: str = ""):
        logger.info("[Agent1] %s — %s", step, message)
        self.progress_callback(step, message)

    def _set_state(self, key: str, value):
        self._state[key] = value

    # ──────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────

    def run(
        self,
        urls: List[str],
        keywords: str = "",
        top_n: int = 10,
    ) -> Dict:
        """
        Execute the full Agent 1 pipeline.

        Returns a result dict:
            {
                "success"       : bool,
                "error"         : str | None,
                "validation"    : {...},
                "ranking"       : {...},
                "analyses"      : [{metadata, scan, statistics, knowledge, report_path}, …],
                "elapsed_sec"   : float,
            }
        """
        t_start = time.time()
        result = {
            "success": False,
            "error": None,
            "validation": {},
            "ranking": {},
            "analyses": [],
            "elapsed_sec": 0.0,
        }

        # ── Step 1: Validate URLs ──────────────────────────────
        self._progress("Validating repositories", f"Received {len(urls)} URLs")
        validation = validate_url_list(urls)
        result["validation"] = validation

        if not validation["valid"]:
            result["error"] = "No valid URLs to process. " + "; ".join(
                e["message"] for e in validation["errors"]
            )
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        valid_urls = validation["valid"]

        # ── Step 2: Collect GitHub metadata ───────────────────
        self._progress("Collecting metadata", f"Fetching info for {len(valid_urls)} repos")
        repos_meta = fetch_repos_metadata(valid_urls)

        # ── Step 3: Keyword filtering & scoring ───────────────
        self._progress("Filtering repositories", f"Keywords: {keywords!r}")
        scored_repos = filter_and_score(repos_meta, keywords)

        # ── Step 4: Rank ──────────────────────────────────────
        self._progress("Ranking repositories", f"Top {top_n} selected")
        ranked = rank_repositories(scored_repos, top_n=top_n)
        summary = get_ranking_summary(ranked, keywords, len(valid_urls))
        result["ranking"] = summary

        if not ranked:
            result["error"] = "No repositories passed relevance filtering."
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        # ── Step 5–9: Per-repository deep analysis ────────────
        analyses = []
        for repo_meta in ranked:
            if repo_meta.get("error"):
                analyses.append({
                    "metadata": repo_meta,
                    "error": repo_meta["error"],
                })
                continue

            owner = repo_meta["owner"]
            name  = repo_meta["name"]
            label = f"{owner}/{name}"

            # Clone
            self._progress("Cloning repository", label)
            clone_res = clone_repository(owner, name, repo_meta.get("clone_url", ""))
            if not clone_res["success"]:
                analyses.append({
                    "metadata": repo_meta,
                    "error": f"Clone failed: {clone_res['message']}",
                })
                continue

            local_path = clone_res["local_path"]

            # Scan
            self._progress("Scanning files", label)
            scan = scan_repository(local_path, repo_name=label)

            # Statistics
            self._progress("Analysing source code", label)
            stats = compute_statistics(scan)

            # Inject code entities back into metadata for possible re-scoring
            repo_meta["code_entities_text"] = stats.get("code_entities_text", "")

            # Knowledge
            self._progress("Building knowledge", label)
            knowledge = generate_knowledge(repo_meta, scan, stats)
            knowledge_path = save_knowledge(knowledge, owner, name)

            # Report
            self._progress("Generating report", label)
            report_text = generate_report(knowledge)
            report_path = save_report(report_text, owner, name)

            analyses.append({
                "metadata":      repo_meta,
                "scan_summary": {
                    "total_files":  scan["total_files"],
                    "source_files": scan["source_files"],
                    "scan_errors":  len(scan["scan_errors"]),
                },
                "statistics":    stats,
                "knowledge":     knowledge,
                "knowledge_path":knowledge_path,
                "report_path":   report_path,
                "clone_info":    clone_res,
                "error":         None,
            })

        result["analyses"] = analyses
        result["success"]  = True
        result["elapsed_sec"] = round(time.time() - t_start, 2)

        self._progress("Agent 1 complete", f"{len(analyses)} repositories analysed in {result['elapsed_sec']}s")
        return result

    # ──────────────────────────────────────────────
    # Lightweight discovery: metadata + filter + rank only (no cloning)
    # ──────────────────────────────────────────────

    def discover(
        self,
        urls: List[str],
        keywords: str = "",
        top_n: int = 10,
    ) -> Dict:
        """
        Fast discovery pipeline: validate → metadata → filter → rank.
        Does NOT clone or scan any repository.

        Returns:
            {
                "success"    : bool,
                "error"      : str | None,
                "validation" : {...},
                "ranking"    : {...},
                "elapsed_sec": float,
            }
        """
        t_start = time.time()
        result = {
            "success": False,
            "error": None,
            "validation": {},
            "ranking": {},
            "elapsed_sec": 0.0,
        }

        # Step 1: Validate
        self._progress("Validating repositories", f"{len(urls)} URLs received")
        validation = validate_url_list(urls)
        result["validation"] = validation

        if not validation["valid"]:
            result["error"] = "No valid URLs. " + "; ".join(
                e["message"] for e in validation["errors"]
            )
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        valid_urls = validation["valid"]

        # Step 2: GitHub metadata
        self._progress("Collecting metadata", f"{len(valid_urls)} repositories")
        repos_meta = fetch_repos_metadata(valid_urls)

        # Step 3: Filter & score
        self._progress("Filtering repositories", f"Keywords: {keywords!r}")
        scored = filter_and_score(repos_meta, keywords)

        # Step 4: Rank
        self._progress("Ranking repositories", f"Top {top_n}")
        ranked = rank_repositories(scored, top_n=top_n)
        summary = get_ranking_summary(ranked, keywords, len(valid_urls))
        result["ranking"] = summary

        result["success"] = True
        result["elapsed_sec"] = round(time.time() - t_start, 2)
        self._progress(
            "Discovery complete",
            f"{len(ranked)} repositories ranked in {result['elapsed_sec']}s",
        )
        return result

    # ──────────────────────────────────────────────
    # Convenience: analyse a single already-cloned repo
    # ──────────────────────────────────────────────

    def analyse_single(self, owner: str, name: str, local_path: str, metadata: Dict) -> Dict:
        """
        Run only the analysis sub-pipeline (scan → stats → knowledge → report)
        for a single repository that has already been cloned.
        """
        self._progress("Scanning files", f"{owner}/{name}")
        scan = scan_repository(local_path, repo_name=f"{owner}/{name}")

        self._progress("Analysing source code", f"{owner}/{name}")
        stats = compute_statistics(scan)

        self._progress("Building knowledge", f"{owner}/{name}")
        knowledge = generate_knowledge(metadata, scan, stats)
        knowledge_path = save_knowledge(knowledge, owner, name)

        self._progress("Generating report", f"{owner}/{name}")
        report_text = generate_report(knowledge)
        report_path = save_report(report_text, owner, name)

        return {
            "scan_summary": {
                "total_files":  scan["total_files"],
                "source_files": scan["source_files"],
                "scan_errors":  len(scan["scan_errors"]),
            },
            "statistics":    stats,
            "knowledge":     knowledge,
            "knowledge_path":knowledge_path,
            "report_path":   report_path,
        }

    # ──────────────────────────────────────────────
    # Agent info
    # ──────────────────────────────────────────────

    @classmethod
    def describe(cls) -> Dict:
        return {
            "name":         cls.AGENT_NAME,
            "version":      cls.AGENT_VERSION,
            "phase":        "Phase 1",
            "status":       "Active",
            "capabilities": [
                "GitHub URL validation",
                "Repository metadata collection",
                "Keyword-based relevance scoring",
                "Top-N repository ranking",
                "Repository cloning (shallow)",
                "Python AST-based code analysis",
                "Java source code analysis (javalang + regex fallback)",
                "Repository statistics computation",
                "Structured JSON knowledge generation",
                "Human-readable Markdown report generation",
            ],
        }
