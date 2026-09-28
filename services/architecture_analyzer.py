"""
Architecture Analyzer (Agent 2 component)
Analyses the CodeGraph to infer architecture layers, key components,
dependency patterns, and potential entry points.

All reasoning is evidence-based — no layer is claimed without supporting data.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

from services.graph_builder import CodeGraph

logger = logging.getLogger(__name__)

# ── Heuristics ────────────────────────────────────────────────

_PRESENTATION_HINTS = {
    "view", "views", "template", "templates", "ui", "frontend",
    "page", "pages", "controller", "controllers", "handler", "handlers",
    "route", "routes", "api", "rest", "graphql", "web", "http",
}
_SERVICE_HINTS = {
    "service", "services", "manager", "managers", "processor",
    "handler", "engine", "usecase", "application", "logic", "business",
}
_DATA_HINTS = {
    "model", "models", "repository", "repositories", "dao", "db",
    "database", "entity", "entities", "schema", "migration", "migrations",
    "store", "storage", "persistence",
}
_ENTRY_HINTS = {
    "main", "app", "run", "start", "index", "__main__",
    "server", "wsgi", "asgi", "manage",
}


def _path_tokens(path: str) -> Set[str]:
    import re
    return set(re.split(r'[/\\._\-]', path.lower()))


def _classify_file(path: str) -> Optional[str]:
    tokens = _path_tokens(path)
    if tokens & _PRESENTATION_HINTS:
        return "Presentation"
    if tokens & _SERVICE_HINTS:
        return "Service"
    if tokens & _DATA_HINTS:
        return "Data"
    return None


def _is_entry_point(path: str) -> bool:
    tokens = _path_tokens(path)
    return bool(tokens & _ENTRY_HINTS)


# ── Main analyser ─────────────────────────────────────────────

class ArchitectureAnalyzer:
    """
    Consumes a CodeGraph and produces an architecture analysis report dict.
    """

    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.G = graph.G

    def analyse(self) -> Dict:
        layers = self._infer_layers()
        key_components = self._find_key_components()
        entry_points = self._find_entry_points()
        cycles = self._find_cycles()
        dependency_clusters = self._find_clusters()
        highly_connected = self._highly_connected_nodes(top_n=10)
        module_deps = self._module_dependency_summary()

        return {
            "repo_name":           self.graph.repo_name,
            "layers":              layers,
            "key_components":      key_components,
            "entry_points":        entry_points,
            "cyclic_dependencies": cycles,
            "dependency_clusters": dependency_clusters,
            "highly_connected":    highly_connected,
            "module_dependencies": module_deps,
            "graph_stats":         self.graph.stats(),
            "architecture_summary": self._build_summary(layers, entry_points, key_components, cycles),
            "repo_diagram":        self.build_repo_diagram(),
        }

    # ── Layer inference ───────────────────────────────────────

    def _infer_layers(self) -> Dict[str, List[str]]:
        layers: Dict[str, List[str]] = {
            "Presentation": [],
            "Service":      [],
            "Data":         [],
            "Unclassified": [],
        }
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") != "File":
                continue
            layer = _classify_file(node)
            layers[layer or "Unclassified"].append(node)

        # Remove empty layers
        return {k: v for k, v in layers.items() if v}

    # ── Entry points ──────────────────────────────────────────

    def _find_entry_points(self) -> List[str]:
        entries = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "File" and _is_entry_point(node):
                entries.append(node)
        return entries

    # ── Key components ────────────────────────────────────────

    def _find_key_components(self, top_n: int = 10) -> List[Dict]:
        """Nodes with highest in-degree (most depended-upon)."""
        scored = []
        for node, data in self.G.nodes(data=True):
            ntype = data.get("node_type", "")
            if ntype in ("File", "Class", "Module"):
                in_deg  = self.G.in_degree(node)
                out_deg = self.G.out_degree(node)
                if in_deg + out_deg > 0:
                    scored.append({
                        "id":       node,
                        "label":    data.get("label", node),
                        "type":     ntype,
                        "in_degree":  in_deg,
                        "out_degree": out_deg,
                        "total":      in_deg + out_deg,
                    })
        scored.sort(key=lambda x: x["total"], reverse=True)
        return scored[:top_n]

    # ── Highly connected ──────────────────────────────────────

    def _highly_connected_nodes(self, top_n: int = 10) -> List[Dict]:
        scored = []
        for node, data in self.G.nodes(data=True):
            deg = self.G.degree(node)
            if deg > 1:
                scored.append({
                    "id":    node,
                    "label": data.get("label", node),
                    "type":  data.get("node_type", "Unknown"),
                    "degree": deg,
                })
        scored.sort(key=lambda x: x["degree"], reverse=True)
        return scored[:top_n]

    # ── Cycles ────────────────────────────────────────────────

    def _find_cycles(self) -> List[List[str]]:
        """Find simple cycles (file-level only to keep output manageable)."""
        file_nodes = {n for n, d in self.G.nodes(data=True) if d.get("node_type") == "File"}
        sub = self.G.subgraph(file_nodes)
        try:
            cycles = list(nx.simple_cycles(sub))
            return [c for c in cycles if len(c) <= 10][:20]
        except Exception:
            return []

    # ── Dependency clusters ───────────────────────────────────

    def _find_clusters(self) -> List[Dict]:
        """Weakly connected components as clusters."""
        file_nodes = {n for n, d in self.G.nodes(data=True) if d.get("node_type") == "File"}
        sub = self.G.subgraph(file_nodes).to_undirected()
        clusters = []
        for comp in sorted(nx.connected_components(sub), key=len, reverse=True):
            if len(comp) > 1:
                clusters.append({"size": len(comp), "members": sorted(comp)[:10]})
        return clusters[:10]

    # ── Module dependency summary ─────────────────────────────

    def _module_dependency_summary(self) -> List[Dict]:
        deps = []
        for u, v, data in self.G.edges(data=True):
            if data.get("relation") in ("IMPORTS", "DEPENDS_ON"):
                u_data = self.G.nodes.get(u, {})
                v_data = self.G.nodes.get(v, {})
                deps.append({
                    "from":     u_data.get("label", u),
                    "to":       v_data.get("label", v),
                    "relation": data.get("relation"),
                })
        return deps[:50]

    # ── Summary text (human-readable narrative) ──────────────

    def _build_summary(self, layers, entry_points, key_components, cycles) -> str:
        """
        Build a short, plain-English summary of the repository architecture.
        No raw graph numbers, node IDs, or technical graph jargon.
        Based entirely on the evidence found in the repository.
        """
        repo = self.graph.repo_name
        sentences = []

        # Opening — what the repo is
        sentences.append(f"{repo} is a software project")

        # Primary language hint from file counts
        py_files  = sum(1 for n, d in self.G.nodes(data=True)
                        if d.get("node_type") == "File" and d.get("language") == "Python")
        java_files = sum(1 for n, d in self.G.nodes(data=True)
                         if d.get("node_type") == "File" and d.get("language") == "Java")
        if py_files and java_files:
            sentences[-1] += f" written primarily in Python and Java"
        elif py_files:
            sentences[-1] += f" written in Python"
        elif java_files:
            sentences[-1] += f" written in Java"
        sentences[-1] += "."

        # Entry points
        if entry_points:
            ep_names = [ep.split("/")[-1] for ep in entry_points[:3]]
            sentences.append(
                f"The application starts from "
                f"{'and '.join(ep_names)}, "
                f"{'which serve' if len(ep_names) > 1 else 'which serves'} as the main entry "
                f"{'points' if len(ep_names) > 1 else 'point'}."
            )

        # Layers — describe what each layer contains
        if layers:
            layer_descs = {
                "Presentation": "handles user-facing features such as routes, views, or API endpoints",
                "Service":      "contains the core business logic and application processing",
                "Data":         "manages data models, database access, and persistence",
                "Unclassified": "contains supporting modules and utilities",
            }
            for layer_name, files in layers.items():
                desc = layer_descs.get(layer_name, "contains supporting code")
                example_files = [f.split("/")[-1] for f in files[:3]]
                sentences.append(
                    f"The {layer_name} layer ({', '.join(example_files)}"
                    f"{' and others' if len(files) > 3 else ''}) {desc}."
                )

        # Key components — describe the most important ones
        if key_components:
            top = key_components[:3]
            top_names = [c["label"] for c in top]
            sentences.append(
                f"The most central components are "
                f"{', '.join(top_names[:-1]) + ' and ' + top_names[-1] if len(top_names) > 1 else top_names[0]}, "
                f"which are used by the most other parts of the codebase."
            )

        # Cyclic dependencies — plain warning if any
        if cycles:
            sentences.append(
                f"Note: {len(cycles)} circular dependency "
                f"{'chains were' if len(cycles) > 1 else 'chain was'} detected, "
                f"which may make the codebase harder to maintain."
            )

        # Module dependency count as a simple closing fact
        import_edges = sum(
            1 for _, _, d in self.G.edges(data=True)
            if d.get("relation") in ("IMPORTS", "DEPENDS_ON")
        )
        if import_edges:
            sentences.append(
                f"Overall, {import_edges} import or dependency relationships were identified "
                f"between the repository's modules."
            )

        return " ".join(sentences)

    # ── Repository-specific diagram data ─────────────────────

    def build_repo_diagram(self) -> List[Dict]:
        """
        Return a minimal list of component nodes and directed edges
        suitable for rendering a simple human-readable diagram.

        Returns: [{"label": str, "children": [{"label": str, "relation": str}]}]
        Only includes File-level nodes with at least one connection.
        Capped at 12 nodes to keep the diagram readable.
        """
        # Collect file nodes ordered by connectivity
        file_nodes = [
            (n, d) for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "File"
        ]
        file_nodes.sort(key=lambda x: self.G.degree(x[0]), reverse=True)
        top_files = {n for n, _ in file_nodes[:12]}

        diagram = []
        seen_pairs: Set[tuple] = set()

        for node in top_files:
            label = self.G.nodes[node].get("label", node)
            # Only the filename, not the full path
            short_label = label.split("/")[-1] if "/" in label else label

            children = []
            for _, target, edata in self.G.out_edges(node, data=True):
                if target not in top_files:
                    continue
                rel = edata.get("relation", "→")
                pair = (node, target)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                t_label = self.G.nodes[target].get("label", target)
                t_short = t_label.split("/")[-1] if "/" in t_label else t_label
                children.append({"label": t_short, "relation": rel})

            diagram.append({"label": short_label, "full_path": node, "children": children})

        return diagram

    # ── Human-readable architecture report (for download) ────

    def build_human_readable_report(self, repo_full_name: str = "") -> str:
        """
        Generate a plain-text architecture report suitable for download.
        Uses readable names and avoids raw node IDs.
        """
        layers        = self._infer_layers()
        key_components= self._find_key_components()
        entry_points  = self._find_entry_points()
        cycles        = self._find_cycles()
        module_deps   = self._module_dependency_summary()
        summary       = self._build_summary(layers, entry_points, key_components, cycles)

        lines = []
        lines.append("=" * 60)
        lines.append(f"Architecture Report — {repo_full_name or self.graph.repo_name}")
        lines.append("=" * 60)
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(summary)
        lines.append("")

        if entry_points:
            lines.append("ENTRY POINTS")
            lines.append("-" * 40)
            for ep in entry_points:
                lines.append(f"  • {ep}")
            lines.append("")

        if layers:
            lines.append("ARCHITECTURAL LAYERS")
            lines.append("-" * 40)
            for layer_name, files in layers.items():
                lines.append(f"  {layer_name} Layer:")
                for f in files[:15]:
                    lines.append(f"    - {f}")
                if len(files) > 15:
                    lines.append(f"    ... and {len(files) - 15} more files")
            lines.append("")

        if key_components:
            lines.append("KEY COMPONENTS")
            lines.append("-" * 40)
            lines.append("  (Listed from most-connected to least)")
            for comp in key_components:
                lines.append(f"  • {comp['label']} ({comp['type']})")
            lines.append("")

        if module_deps:
            lines.append("MODULE RELATIONSHIPS")
            lines.append("-" * 40)
            lines.append("  (Files and modules that depend on each other)")
            for dep in module_deps[:30]:
                lines.append(f"  {dep['from']}  →[{dep['relation']}]→  {dep['to']}")
            if len(module_deps) > 30:
                lines.append(f"  ... and {len(module_deps) - 30} more relationships")
            lines.append("")

        if cycles:
            lines.append("CIRCULAR DEPENDENCIES DETECTED")
            lines.append("-" * 40)
            lines.append("  The following files form circular import chains:")
            for cycle in cycles[:10]:
                lines.append(f"  {' → '.join(cycle)} → (back to start)")
            lines.append("")

        lines.append("=" * 60)
        lines.append("Generated by CodeLens AI — Semantic Code Graph Agent")
        lines.append("=" * 60)
        return "\n".join(lines)
