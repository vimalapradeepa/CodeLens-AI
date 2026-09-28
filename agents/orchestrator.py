"""
Multi-Agent Orchestrator
=========================
Coordinates Agent 1 (Repository Analysis) and Agent 2 (Semantic Code Graph)
into a unified pipeline and exposes a shared knowledge layer for RAG.

Architecture:
    CodeLens AI
         |
    Orchestrator
         |
    ┌────┴────┐
  Agent1   Agent2
    |         |
  Knowledge  Graph/Arch
    └────┬────┘
    SharedKnowledge → RAG → LLM
"""

import logging
import time
from typing import Dict, List, Optional, Callable

from agents.repository_agent   import RepositoryAnalysisAgent
from agents.architecture_agent import SemanticCodeGraphAgent

logger = logging.getLogger(__name__)


class SharedKnowledge:
    """
    Holds the combined output of both agents for a single repository.
    This is the object consumed by the RAG system.
    """

    def __init__(self):
        self.repositories: Dict[str, Dict] = {}   # full_name → combined data

    def store(self, full_name: str, agent1_result: Dict, agent2_result: Dict):
        self.repositories[full_name] = {
            "full_name":    full_name,
            "agent1":       agent1_result,
            "agent2":       agent2_result,
            "combined_at":  time.time(),
        }

    def get(self, full_name: str) -> Optional[Dict]:
        return self.repositories.get(full_name)

    def all_repos(self) -> List[str]:
        return list(self.repositories.keys())

    def load_from_disk(self):
        """
        Scan the output/ directory and reload any knowledge.json files that
        were saved by a previous analysis session.  This lets the QA engine
        answer questions even after a server restart, without re-running analysis.
        """
        import os, json
        from config import OUTPUT_DIR

        if not os.path.isdir(OUTPUT_DIR):
            return

        for entry in os.scandir(OUTPUT_DIR):
            if not entry.is_dir():
                continue
            k_path = os.path.join(entry.path, "knowledge.json")
            a_path = os.path.join(entry.path, "architecture.json")
            if not os.path.exists(k_path):
                continue

            # Derive full_name from directory name (owner__repo → owner/repo)
            dir_name  = entry.name
            full_name = dir_name.replace("__", "/", 1)

            if full_name in self.repositories:
                continue   # already loaded in-memory, skip

            try:
                with open(k_path, "r", encoding="utf-8") as fh:
                    knowledge = json.load(fh)
            except Exception:
                continue

            arch = {}
            if os.path.exists(a_path):
                try:
                    with open(a_path, "r", encoding="utf-8") as fh:
                        arch = json.load(fh)
                except Exception:
                    pass

            repo_info = knowledge.get("repository", {})
            a1_summary = {
                "metadata": {
                    "owner":            repo_info.get("owner", ""),
                    "name":             repo_info.get("name", ""),
                    "full_name":        repo_info.get("full_name", full_name),
                    "description":      repo_info.get("description", ""),
                    "primary_language": repo_info.get("primary_language", ""),
                    "topics":           repo_info.get("topics", []),
                    "url":              repo_info.get("url", ""),
                },
                "statistics": knowledge.get("statistics", {}),
                "knowledge":  knowledge,
                "report_path": os.path.join(entry.path, "report.md"),
            }
            a2_summary = {"architecture": arch}

            self.repositories[full_name] = {
                "full_name":   full_name,
                "agent1":      a1_summary,
                "agent2":      a2_summary,
                "combined_at": 0,   # sentinel: loaded from disk
            }
            logger.info("SharedKnowledge: loaded %s from disk", full_name)

    def flat_text_for_rag(self, full_name: str) -> str:
        """
        Produce a rich plain-text representation of all available knowledge
        about `full_name` suitable for chunking and embedding.

        Emits granular lines (one concept per line) so that keyword and
        semantic retrieval can match specific questions about files, classes,
        methods, functions, imports, dependencies, and architecture.
        """
        data = self.repositories.get(full_name)
        if not data:
            return ""

        parts = []

        # ── Agent 1 knowledge ─────────────────────────────────
        a1 = data.get("agent1", {})
        knowledge = a1.get("knowledge", {})
        repo_info = knowledge.get("repository", {})

        if repo_info:
            parts.append(f"Repository: {repo_info.get('full_name', '')}")
            parts.append(f"Description: {repo_info.get('description', '')}")
            topics = repo_info.get("topics", [])
            if topics:
                parts.append(f"Topics: {', '.join(topics)}")
            parts.append(f"Primary language: {repo_info.get('primary_language', '')}")
            langs = repo_info.get("languages", {})
            if langs:
                parts.append(f"Languages used: {', '.join(langs.keys())}")

        stats = knowledge.get("statistics", {})
        if stats:
            parts.append(
                f"Statistics: {stats.get('total_files')} total files, "
                f"{stats.get('source_files')} source files, "
                f"{stats.get('python_files')} Python files, "
                f"{stats.get('java_files')} Java files, "
                f"{stats.get('classes')} classes, "
                f"{stats.get('methods')} methods, "
                f"{stats.get('functions')} functions, "
                f"{stats.get('imports')} imports."
            )
            deps = stats.get("dependencies", [])
            if deps:
                parts.append(f"Detected dependencies: {', '.join(deps)}")

        # ── Per-file detail (granular — one concept per line) ──
        for f in knowledge.get("files", []):
            if f.get("error") and not f.get("classes") and not f.get("functions"):
                continue

            fpath = f.get("path", "")
            lang  = f.get("language", "")
            parts.append(f"File: {fpath} (language: {lang})")

            # Imports
            imports = f.get("imports", [])
            if imports:
                parts.append(f"File {fpath} imports: {', '.join(imports[:30])}")

            if lang == "Python":
                # Classes with methods
                for cls in f.get("classes", []):
                    if not isinstance(cls, dict):
                        continue
                    cname   = cls.get("name", "")
                    methods = cls.get("methods", [])
                    bases   = cls.get("bases", [])
                    base_str = f" inherits from {', '.join(bases)}" if bases else ""
                    parts.append(
                        f"Class {cname} in file {fpath}{base_str}."
                    )
                    if methods:
                        parts.append(
                            f"Class {cname} has methods: {', '.join(methods)}."
                        )
                        # Each method individually for fine-grained retrieval
                        for m in methods:
                            parts.append(
                                f"Method {m} belongs to class {cname} in file {fpath}."
                            )

                # Standalone functions
                for fn in f.get("functions", []):
                    parts.append(f"Function {fn} defined in file {fpath}.")

                # Calls (relationship signal)
                calls = f.get("calls", [])
                if calls:
                    parts.append(
                        f"File {fpath} makes calls to: {', '.join(list(calls)[:20])}."
                    )

            elif lang == "Java":
                pkg = f.get("package", "")
                if pkg:
                    parts.append(f"File {fpath} belongs to package {pkg}.")
                for cls in f.get("classes", []):
                    if not isinstance(cls, dict):
                        continue
                    cname   = cls.get("name", "")
                    methods = cls.get("methods", [])
                    ext     = cls.get("extends")
                    impls   = cls.get("implements", [])
                    rel_str = ""
                    if ext:
                        rel_str += f" extends {ext}"
                    if impls:
                        rel_str += f" implements {', '.join(impls)}"
                    parts.append(f"Class {cname} in file {fpath}{rel_str}.")
                    if methods:
                        parts.append(
                            f"Class {cname} has methods: {', '.join(methods)}."
                        )
                        for m in methods:
                            parts.append(
                                f"Method {m} belongs to class {cname} in file {fpath}."
                            )
                for iface in f.get("interfaces", []):
                    if isinstance(iface, dict):
                        iname   = iface.get("name", "")
                        methods = iface.get("methods", [])
                        parts.append(f"Interface {iname} defined in file {fpath}.")
                        if methods:
                            parts.append(
                                f"Interface {iname} has methods: {', '.join(methods)}."
                            )

        # ── Agent 2 architecture ──────────────────────────────
        a2   = data.get("agent2", {})
        arch = a2.get("architecture", {})

        if arch.get("architecture_summary"):
            parts.append("--- Architecture Summary ---")
            parts.append(arch["architecture_summary"])

        for comp in arch.get("key_components", []):
            parts.append(
                f"Key component: {comp['label']} (type: {comp['type']}, "
                f"connectivity degree: {comp['total']})."
            )

        for ep in arch.get("entry_points", []):
            parts.append(f"Entry point file: {ep}.")

        for layer_name, layer_files in arch.get("layers", {}).items():
            for lf in layer_files:
                parts.append(f"File {lf} is part of the {layer_name} layer.")

        for dep in arch.get("module_dependencies", [])[:50]:
            parts.append(
                f"Module dependency: {dep.get('from')} {dep.get('relation')} {dep.get('to')}."
            )

        cycles = arch.get("cyclic_dependencies", [])
        if cycles:
            for cycle in cycles[:10]:
                parts.append(f"Cyclic dependency detected: {' -> '.join(cycle)}.")

        return "\n".join(parts)


# ── Orchestrator ──────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Orchestrates Agent 1 → Agent 2 and populates SharedKnowledge.

    Usage::

        orch = MultiAgentOrchestrator()
        result = orch.run_full_pipeline(urls=[...], keywords="...")
        shared = orch.shared_knowledge
    """

    def __init__(self, progress_callback: Optional[Callable[[str, str], None]] = None):
        self.progress_callback = progress_callback or (lambda s, m: None)
        self.shared_knowledge = SharedKnowledge()
        self._agent1 = RepositoryAnalysisAgent(
            progress_callback=lambda s, m: self.progress_callback(f"[Agent1] {s}", m)
        )
        self._agent2 = SemanticCodeGraphAgent(
            progress_callback=lambda s, m: self.progress_callback(f"[Agent2] {s}", m)
        )

    def _progress(self, step: str, msg: str = ""):
        logger.info("[Orchestrator] %s — %s", step, msg)
        self.progress_callback(step, msg)

    # ──────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────

    def run_full_pipeline(self, urls: List[str], keywords: str = "", top_n: int = 10) -> Dict:
        """
        Run Agent 1 then Agent 2 for every successfully analysed repository.

        Returns:
            {
                "success"        : bool,
                "agent1_result"  : dict (full Agent 1 output),
                "per_repo"       : {full_name: {agent1: …, agent2: …}},
                "elapsed_sec"    : float,
            }
        """
        t_start = time.time()
        result = {
            "success": False,
            "agent1_result": {},
            "per_repo": {},
            "elapsed_sec": 0.0,
        }

        # ── Phase 1: Agent 1 ──────────────────────────────────
        self._progress("Starting Agent 1 — Repository Analysis")
        agent1_output = self._agent1.run(urls=urls, keywords=keywords, top_n=top_n)
        result["agent1_result"] = agent1_output

        if not agent1_output.get("success"):
            result["elapsed_sec"] = round(time.time() - t_start, 2)
            return result

        # ── Phase 2: Agent 2 per repo ─────────────────────────
        self._progress("Starting Agent 2 — Semantic Code Graph")
        for analysis in agent1_output.get("analyses", []):
            if analysis.get("error"):
                continue

            meta = analysis.get("metadata", {})
            owner = meta.get("owner", "")
            name  = meta.get("name", "")
            full_name = meta.get("full_name", f"{owner}/{name}")
            knowledge = analysis.get("knowledge")

            agent2_output = self._agent2.run(owner=owner, name=name, knowledge=knowledge)

            # Build agent1 per-repo summary for storage
            a1_summary = {
                "metadata":   meta,
                "statistics": analysis.get("statistics", {}),
                "knowledge":  knowledge,
                "report_path":analysis.get("report_path", ""),
            }

            self.shared_knowledge.store(full_name, a1_summary, agent2_output)
            result["per_repo"][full_name] = {
                "agent1": a1_summary,
                "agent2": agent2_output,
            }

        result["success"]     = True
        result["elapsed_sec"] = round(time.time() - t_start, 2)
        self._progress(
            "Orchestration complete",
            f"{len(result['per_repo'])} repos processed in {result['elapsed_sec']}s"
        )
        return result

    # ──────────────────────────────────────────────
    # Run agents individually (for UI buttons)
    # ──────────────────────────────────────────────

    def run_discovery(self, urls: List[str], keywords: str = "", top_n: int = 10) -> Dict:
        """
        Lightweight discovery: metadata → filter → rank only. No cloning.
        Used by the 'Find & Analyze' button on the home page.
        """
        self._progress("Running discovery (no cloning)")
        return self._agent1.discover(urls=urls, keywords=keywords, top_n=top_n)

    def run_agent1_only(self, urls: List[str], keywords: str = "", top_n: int = 10) -> Dict:
        self._progress("Running Agent 1 only")
        return self._agent1.run(urls=urls, keywords=keywords, top_n=top_n)

    def run_agent2_for_repo(self, owner: str, name: str, knowledge: Optional[Dict] = None) -> Dict:
        self._progress("Running Agent 2 only", f"{owner}/{name}")
        return self._agent2.run(owner=owner, name=name, knowledge=knowledge)

    # ──────────────────────────────────────────────
    # Agent descriptors (for the Agents page)
    # ──────────────────────────────────────────────

    @staticmethod
    def agent_descriptions() -> List[Dict]:
        return [
            RepositoryAnalysisAgent.describe(),
            SemanticCodeGraphAgent.describe(),
        ]
