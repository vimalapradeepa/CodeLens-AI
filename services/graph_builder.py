"""
Graph Builder (Agent 2 component)
Constructs a semantic code relationship graph from Agent 1 knowledge.

Node types  : Repository, File, Class, Method, Function, Package
Edge types  : CONTAINS, IMPORTS, CALLS, INHERITS, DEFINES, DEPENDS_ON

Uses NetworkX internally; exports to JSON and PyVis HTML.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import networkx as nx

from config import GRAPH_DIR

logger = logging.getLogger(__name__)


class CodeGraph:
    """
    Wraps a NetworkX DiGraph and provides domain-level add_* helpers.
    """

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.G: nx.DiGraph = nx.DiGraph()
        self._add_node(repo_name, node_type="Repository", label=repo_name)

    # ── Node/edge primitives ──────────────────────────────────

    def _add_node(self, node_id: str, **attrs):
        if not self.G.has_node(node_id):
            self.G.add_node(node_id, **attrs)

    def _add_edge(self, src: str, dst: str, rel: str):
        if src and dst and src != dst:
            self.G.add_edge(src, dst, relation=rel)

    # ── Domain helpers ────────────────────────────────────────

    def add_file(self, file_path: str, language: str):
        self._add_node(file_path, node_type="File", label=file_path, language=language)
        self._add_edge(self.repo_name, file_path, "CONTAINS")

    def add_class(self, file_path: str, class_name: str, bases: Optional[List[str]] = None):
        cid = f"{file_path}::{class_name}"
        self._add_node(cid, node_type="Class", label=class_name, file=file_path)
        self._add_edge(file_path, cid, "DEFINES")
        for base in (bases or []):
            # Try to resolve base to a known class node
            base_id = self._find_class_node(base)
            if base_id:
                self._add_edge(cid, base_id, "INHERITS")

    def add_method(self, file_path: str, class_name: str, method_name: str):
        cid  = f"{file_path}::{class_name}"
        mid  = f"{cid}::{method_name}"
        self._add_node(mid, node_type="Method", label=method_name, class_node=cid)
        self._add_edge(cid, mid, "CONTAINS")

    def add_function(self, file_path: str, func_name: str):
        fid = f"{file_path}::{func_name}"
        self._add_node(fid, node_type="Function", label=func_name, file=file_path)
        self._add_edge(file_path, fid, "DEFINES")

    def add_import(self, file_path: str, module_name: str):
        # Module node
        mod_id = f"module::{module_name}"
        self._add_node(mod_id, node_type="Module", label=module_name)
        self._add_edge(file_path, mod_id, "IMPORTS")

    def add_package(self, file_path: str, package_name: str):
        pkg_id = f"package::{package_name}"
        self._add_node(pkg_id, node_type="Package", label=package_name)
        self._add_edge(file_path, pkg_id, "DEPENDS_ON")

    def add_call(self, caller_file: str, callee_name: str):
        """Add a CALLS edge from a file to a method/function node (best-effort)."""
        target = self._find_method_or_func(callee_name)
        if target:
            self._add_edge(caller_file, target, "CALLS")

    # ── Internal resolvers ────────────────────────────────────

    def _find_class_node(self, class_name: str) -> Optional[str]:
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "Class" and data.get("label") == class_name:
                return node
        return None

    def _find_method_or_func(self, name: str) -> Optional[str]:
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") in ("Method", "Function") and data.get("label") == name:
                return node
        return None

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> Dict:
        node_types: Dict[str, int] = {}
        edge_types: Dict[str, int] = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("node_type", "Unknown")
            node_types[t] = node_types.get(t, 0) + 1
        for _, _, data in self.G.edges(data=True):
            r = data.get("relation", "UNKNOWN")
            edge_types[r] = edge_types.get(r, 0) + 1
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types":  node_types,
            "edge_types":  edge_types,
        }

    # ── Export ────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        nodes = [
            {"id": n, **data}
            for n, data in self.G.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "relation": d.get("relation", "")}
            for u, v, d in self.G.edges(data=True)
        ]
        return {"repo": self.repo_name, "nodes": nodes, "edges": edges, "stats": self.stats()}

    def save_json(self, owner: str, name: str) -> str:
        out_dir = os.path.join(GRAPH_DIR, f"{owner}__{name}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "graph.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Graph JSON saved → %s", path)
        return path

    def save_html(self, owner: str, name: str) -> str:
        """Generate interactive PyVis HTML visualisation."""
        out_dir = os.path.join(GRAPH_DIR, f"{owner}__{name}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "graph.html")

        try:
            from pyvis.network import Network

            COLOR_MAP = {
                "Repository": "#e74c3c",
                "File":       "#3498db",
                "Class":      "#2ecc71",
                "Method":     "#f39c12",
                "Function":   "#9b59b6",
                "Module":     "#1abc9c",
                "Package":    "#e67e22",
            }

            net = Network(
                height="700px", width="100%",
                directed=True,
                bgcolor="#1a1a2e",
                font_color="#ffffff",
            )
            net.set_options("""
            {
              "physics": {"stabilization": {"iterations": 100}},
              "edges":   {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}},
              "nodes":   {"font": {"size": 12}}
            }
            """)

            for node_id, data in self.G.nodes(data=True):
                ntype = data.get("node_type", "Unknown")
                color = COLOR_MAP.get(ntype, "#95a5a6")
                net.add_node(
                    node_id,
                    label=data.get("label", node_id)[:30],
                    color=color,
                    title=f"{ntype}: {data.get('label', node_id)}",
                )

            for u, v, data in self.G.edges(data=True):
                net.add_edge(u, v, title=data.get("relation", ""), label=data.get("relation", ""))

            net.save_graph(path)
            logger.info("Graph HTML saved → %s", path)

        except ImportError:
            # PyVis not available — write a minimal placeholder
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("<html><body><p>PyVis not installed. Install pyvis to enable graph visualisation.</p></body></html>")
            logger.warning("PyVis not available; placeholder HTML written.")

        return path


# ── Public factory ────────────────────────────────────────────

def build_graph_from_knowledge(knowledge: Dict) -> CodeGraph:
    """
    Build a CodeGraph from an Agent 1 knowledge dict.
    This is the primary entry point for Agent 2.
    """
    repo_info = knowledge.get("repository", {})
    repo_name = repo_info.get("full_name") or repo_info.get("name", "unknown")
    graph = CodeGraph(repo_name)

    files = knowledge.get("files", [])

    for file_data in files:
        if file_data.get("error") and not file_data.get("classes") and not file_data.get("functions"):
            continue

        fpath = file_data.get("path", "")
        lang  = file_data.get("language", "Unknown")
        graph.add_file(fpath, lang)

        # Imports
        for imp in file_data.get("imports", []):
            graph.add_import(fpath, imp)

        if lang == "Python":
            # Classes
            for cls in file_data.get("classes", []):
                if not isinstance(cls, dict):
                    continue
                cname = cls.get("name", "")
                bases = cls.get("bases", [])
                graph.add_class(fpath, cname, bases)
                for method in cls.get("methods", []):
                    graph.add_method(fpath, cname, method)
            # Standalone functions
            for func in file_data.get("functions", []):
                graph.add_function(fpath, func)
            # Calls
            for call in file_data.get("calls", []):
                graph.add_call(fpath, call)

        elif lang == "Java":
            # Package
            pkg = file_data.get("package", "")
            if pkg:
                graph.add_package(fpath, pkg)
            # Classes
            for cls in file_data.get("classes", []):
                if not isinstance(cls, dict):
                    continue
                cname = cls.get("name", "")
                bases = [cls["extends"]] if cls.get("extends") else []
                graph.add_class(fpath, cname, bases)
                for method in cls.get("methods", []):
                    graph.add_method(fpath, cname, method)

    return graph
