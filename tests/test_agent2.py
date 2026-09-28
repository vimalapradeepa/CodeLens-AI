"""Tests for Agent 2 — Semantic Code Graph and Architecture Agent."""
import pytest
from services.graph_builder import CodeGraph, build_graph_from_knowledge
from services.architecture_analyzer import ArchitectureAnalyzer
from agents.architecture_agent import SemanticCodeGraphAgent


SAMPLE_KNOWLEDGE = {
    "repository": {"name": "sample", "full_name": "user/sample", "description": "Test repo"},
    "statistics": {"total_files": 3, "classes": 2, "methods": 4, "functions": 2},
    "files": [
        {
            "path": "src/models.py",
            "language": "Python",
            "classes": [{"name": "User", "methods": ["login", "logout"], "bases": []}],
            "functions": ["get_user"],
            "imports": ["database", "flask"],
            "calls": ["database.connect"],
            "error": None,
        },
        {
            "path": "src/services.py",
            "language": "Python",
            "classes": [{"name": "AuthService", "methods": ["authenticate"], "bases": ["BaseService"]}],
            "functions": [],
            "imports": ["models"],
            "calls": ["User.login"],
            "error": None,
        },
        {
            "path": "main.py",
            "language": "Python",
            "classes": [],
            "functions": ["main"],
            "imports": ["services"],
            "calls": ["AuthService.authenticate"],
            "error": None,
        },
    ],
}


class TestCodeGraph:

    def test_build_from_knowledge(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        stats = graph.stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    def test_node_types_present(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        stats = graph.stats()
        node_types = stats["node_types"]
        assert "File" in node_types
        assert "Class" in node_types

    def test_edge_types_present(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        stats = graph.stats()
        edge_types = stats["edge_types"]
        assert "CONTAINS" in edge_types or "DEFINES" in edge_types

    def test_to_dict(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) > 0

    def test_repo_node_exists(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        repo_nodes = [n for n, d in graph.G.nodes(data=True) if d.get("node_type") == "Repository"]
        assert len(repo_nodes) == 1


class TestArchitectureAnalyzer:

    def test_analyse_returns_dict(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        analyser = ArchitectureAnalyzer(graph)
        result = analyser.analyse()
        assert isinstance(result, dict)
        assert "key_components" in result
        assert "entry_points" in result
        assert "architecture_summary" in result

    def test_entry_point_detected(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        analyser = ArchitectureAnalyzer(graph)
        result = analyser.analyse()
        # main.py should be detected as an entry point
        assert any("main" in ep.lower() for ep in result["entry_points"])

    def test_summary_not_empty(self):
        graph = build_graph_from_knowledge(SAMPLE_KNOWLEDGE)
        analyser = ArchitectureAnalyzer(graph)
        result = analyser.analyse()
        assert len(result["architecture_summary"]) > 10


class TestAgent2:

    def test_describe(self):
        desc = SemanticCodeGraphAgent.describe()
        assert desc["name"] == "Semantic Code Graph Agent"
        assert desc["phase"] == "Phase 2"

    def test_run_with_knowledge(self, tmp_path):
        import sys, os
        # Redirect output to tmp_path
        import config
        orig_graph = config.GRAPH_DIR
        orig_out   = config.OUTPUT_DIR
        config.GRAPH_DIR  = str(tmp_path / "graphs")
        config.OUTPUT_DIR = str(tmp_path / "output")
        os.makedirs(config.GRAPH_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        agent = SemanticCodeGraphAgent()
        result = agent.run(owner="user", name="sample", knowledge=SAMPLE_KNOWLEDGE)

        config.GRAPH_DIR  = orig_graph
        config.OUTPUT_DIR = orig_out

        assert result["success"] is True
        assert result["graph_stats"]["total_nodes"] > 0
