"""
Agent 2 — Semantic Code Graph and Architecture Agent
======================================================
Custom agent built from scratch for CodeLens AI.

Responsibility:
  - Consume Agent 1 structured knowledge
  - Build a semantic code relationship graph
  - Analyse architecture layers, entry points, key components
  - Detect cyclic dependencies
  - Generate an architecture summary
  - Persist graph JSON + interactive HTML

This agent does NOT depend on CrewAI, AutoGen, or LangGraph.
All graph and analysis logic is implemented in services/.
"""

import os
import json
import logging
import time
from typing import Dict, Optional, Callable

from services.graph_builder import build_graph_from_knowledge
from services.architecture_analyzer import ArchitectureAnalyzer
from services.knowledge_generator import load_knowledge
from config import GRAPH_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


class SemanticCodeGraphAgent:
    """
    Agent 2: Semantic Code Graph and Architecture Agent

    Usage::

        agent = SemanticCodeGraphAgent()
        result = agent.run(owner="example", name="repo")
    """

    AGENT_NAME = "Semantic Code Graph Agent"
    AGENT_VERSION = "1.0.0"

    def __init__(self, progress_callback: Optional[Callable[[str, str], None]] = None):
        self.progress_callback = progress_callback or (lambda s, m: None)

    def _progress(self, step: str, msg: str = ""):
        logger.info("[Agent2] %s — %s", step, msg)
        self.progress_callback(step, msg)

    # ──────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────

    def run(self, owner: str, name: str, knowledge: Optional[Dict] = None) -> Dict:
        """
        Build graph + architecture analysis for a repository.

        If `knowledge` is None, loads from the output directory (requires Agent 1 ran first).

        Returns:
            {
                "success"      : bool,
                "error"        : str | None,
                "graph_stats"  : dict,
                "graph_json_path" : str,
                "graph_html_path" : str,
                "architecture" : dict,
                "elapsed_sec"  : float,
            }
        """
        t_start = time.time()
        result = {
            "success": False,
            "error": None,
            "graph_stats": {},
            "graph_json_path": "",
            "graph_html_path": "",
            "architecture": {},
            "elapsed_sec": 0.0,
        }

        # ── Load knowledge ──────────────────────────────────
        self._progress("Loading knowledge", f"{owner}/{name}")
        if knowledge is None:
            knowledge = load_knowledge(owner, name)
        if not knowledge:
            result["error"] = f"No knowledge found for {owner}/{name}. Run Agent 1 first."
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        # ── Build graph ──────────────────────────────────────
        self._progress("Building semantic graph", f"{owner}/{name}")
        try:
            graph = build_graph_from_knowledge(knowledge)
        except Exception as exc:
            result["error"] = f"Graph construction failed: {exc}"
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        graph_stats = graph.stats()
        result["graph_stats"] = graph_stats

        # ── Save graph ───────────────────────────────────────
        self._progress("Saving graph", f"{owner}/{name}")
        json_path = graph.save_json(owner, name)
        html_path = graph.save_html(owner, name)
        result["graph_json_path"] = json_path
        result["graph_html_path"] = html_path

        # ── Architecture analysis ────────────────────────────
        self._progress("Analysing architecture", f"{owner}/{name}")
        try:
            analyser = ArchitectureAnalyzer(graph)
            arch = analyser.analyse()
            result["architecture"] = arch
            self._save_architecture(arch, owner, name)
        except Exception as exc:
            logger.warning("Architecture analysis failed: %s", exc)
            result["architecture"] = {"error": str(exc)}

        result["success"]     = True
        result["elapsed_sec"] = round(time.time() - t_start, 2)
        self._progress("Agent 2 complete", f"Graph: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        return result

    def _save_architecture(self, arch: Dict, owner: str, name: str):
        out_dir = os.path.join(OUTPUT_DIR, f"{owner}__{name}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "architecture.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(arch, fh, indent=2, default=str)
        logger.info("Architecture analysis saved → %s", path)

    # ──────────────────────────────────────────────
    # Load previously generated results
    # ──────────────────────────────────────────────

    @staticmethod
    def load_results(owner: str, name: str) -> Dict:
        """Load cached Agent 2 results from disk."""
        out_dir = os.path.join(OUTPUT_DIR, f"{owner}__{name}")
        arch_path = os.path.join(out_dir, "architecture.json")
        graph_json = os.path.join(GRAPH_DIR, f"{owner}__{name}", "graph.json")
        graph_html = os.path.join(GRAPH_DIR, f"{owner}__{name}", "graph.html")

        arch = {}
        if os.path.exists(arch_path):
            with open(arch_path, "r", encoding="utf-8") as fh:
                arch = json.load(fh)

        return {
            "architecture":    arch,
            "graph_json_path": graph_json if os.path.exists(graph_json) else "",
            "graph_html_path": graph_html if os.path.exists(graph_html) else "",
        }

    # ──────────────────────────────────────────────
    # Agent info
    # ──────────────────────────────────────────────

    @classmethod
    def describe(cls) -> Dict:
        return {
            "name":         cls.AGENT_NAME,
            "version":      cls.AGENT_VERSION,
            "phase":        "Phase 2",
            "status":       "Active",
            "capabilities": [
                "Semantic code relationship graph construction",
                "Node types: Repository, File, Class, Method, Function, Module, Package",
                "Edge types: CONTAINS, IMPORTS, CALLS, INHERITS, DEFINES, DEPENDS_ON",
                "Architecture layer inference (Presentation / Service / Data)",
                "Entry point detection",
                "Key component identification (by graph centrality)",
                "Cyclic dependency detection",
                "Dependency cluster analysis",
                "Interactive HTML graph visualisation (PyVis)",
                "Architecture summary generation",
            ],
        }
