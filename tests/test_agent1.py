"""Integration tests for Agent 1 (without network/git calls)."""
import os, tempfile, pytest
from unittest.mock import patch, MagicMock
from agents.repository_agent import RepositoryAnalysisAgent


def _fake_meta(url, owner="user", name="repo", desc="A python ML library", topics=None, lang="Python"):
    return {
        "url": url, "owner": owner, "name": name,
        "full_name": f"{owner}/{name}",
        "description": desc,
        "topics": topics or ["machine-learning"],
        "primary_language": lang,
        "languages": {lang: 100},
        "stars": 100, "forks": 10, "size_kb": 500,
        "default_branch": "main", "readme": "ML library with RAG support",
        "clone_url": f"https://github.com/{owner}/{name}.git",
        "created_at": "", "updated_at": "", "error": None,
    }


def _make_py_repo(tmp):
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    with open(os.path.join(src, "model.py"), "w") as f:
        f.write("class Model:\n    def predict(self):\n        pass\n")
    with open(os.path.join(src, "utils.py"), "w") as f:
        f.write("def load_data(path):\n    pass\n")
    return tmp


class TestAgent1Integration:

    @patch("agents.repository_agent.fetch_repos_metadata")
    @patch("agents.repository_agent.clone_repository")
    def test_full_pipeline_mock(self, mock_clone, mock_meta):
        with tempfile.TemporaryDirectory() as tmp:
            _make_py_repo(tmp)

            urls = [
                f"https://github.com/user/repo{i}" for i in range(1, 6)
            ]
            metas = [_fake_meta(u, "user", f"repo{i}") for i, u in enumerate(urls, 1)]
            mock_meta.return_value = metas
            mock_clone.return_value = {"success": True, "local_path": tmp, "message": "ok", "already_existed": True}

            agent = RepositoryAnalysisAgent()
            result = agent.run(urls=urls, keywords="machine learning")

            assert result["success"] is True
            assert len(result["analyses"]) > 0
            first = result["analyses"][0]
            assert first["error"] is None
            assert first["statistics"]["python_files"] >= 1

    def test_url_validation_too_few(self):
        # Only 1 URL → validate_url_list should report an error and return no valid URLs
        # We patch fetch_repos_metadata so no network call is made
        with patch("agents.repository_agent.fetch_repos_metadata", return_value=[]):
            agent = RepositoryAnalysisAgent()
            result = agent.run(urls=["https://github.com/a/b"], keywords="test")
            # Either fails because < 5 valid, or because no repos passed
            assert result["success"] is False or result.get("error") is not None

    def test_describe(self):
        desc = RepositoryAnalysisAgent.describe()
        assert desc["name"] == "Repository Analysis Agent"
        assert desc["phase"] == "Phase 1"
        assert len(desc["capabilities"]) > 0


class TestAnalyseSingleDashboardDataFlow:
    """
    Verify that analyse_single returns a dict whose shape matches what
    the dashboard template expects when it accesses data.agent1.statistics
    and data.agent1.knowledge.
    """

    def test_analyse_single_result_keys(self):
        """analyse_single must return statistics and knowledge at top level."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_py_repo(tmp)
            agent = RepositoryAnalysisAgent()
            meta  = _fake_meta("https://github.com/user/repo", "user", "repo")
            result = agent.analyse_single(
                owner="user", name="repo",
                local_path=tmp, metadata=meta,
            )
            assert "statistics" in result
            assert "knowledge"  in result
            assert "report_path" in result
            assert isinstance(result["statistics"], dict)
            assert isinstance(result["knowledge"],  dict)

    def test_a1_summary_shape_matches_dashboard(self):
        """
        The a1_summary dict built in api_analyze must give the dashboard
        template access to data.agent1.statistics and data.agent1.knowledge.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _make_py_repo(tmp)
            agent = RepositoryAnalysisAgent()
            meta  = _fake_meta("https://github.com/user/repo", "user", "repo")
            a1_result = agent.analyse_single(
                owner="user", name="repo",
                local_path=tmp, metadata=meta,
            )
            # Build the same a1_summary that api_analyze builds
            a1_summary = {
                "metadata":    meta,
                "statistics":  a1_result.get("statistics", {}),
                "knowledge":   a1_result.get("knowledge"),
                "report_path": a1_result.get("report_path", ""),
            }
            # Simulate SharedKnowledge.store() output
            data = {
                "full_name":   "user/repo",
                "agent1":      a1_summary,
                "agent2":      {},
                "combined_at": 0,
            }
            # Template accesses: data.agent1.statistics and data.agent1.knowledge
            assert data["agent1"]["statistics"] is not None
            assert data["agent1"]["knowledge"]  is not None
            assert data["agent1"]["statistics"].get("total_files") is not None

    def test_shared_knowledge_store_and_get(self):
        """SharedKnowledge.store then .get must return the same agent1 data."""
        from agents.orchestrator import SharedKnowledge
        with tempfile.TemporaryDirectory() as tmp:
            _make_py_repo(tmp)
            agent = RepositoryAnalysisAgent()
            meta  = _fake_meta("https://github.com/user/repo", "user", "repo")
            a1_result = agent.analyse_single(
                owner="user", name="repo",
                local_path=tmp, metadata=meta,
            )
            a1_summary = {
                "metadata":    meta,
                "statistics":  a1_result["statistics"],
                "knowledge":   a1_result["knowledge"],
                "report_path": a1_result["report_path"],
            }
            sk = SharedKnowledge()
            sk.store("user/repo", a1_summary, {})

            retrieved = sk.get("user/repo")
            assert retrieved is not None
            assert retrieved["agent1"]["statistics"] == a1_summary["statistics"]
            assert retrieved["agent1"]["knowledge"]  == a1_summary["knowledge"]

    def test_pipeline_result_per_repo_shape(self):
        """_pipeline_result['per_repo'][full_name] must have agent1.statistics accessible."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_py_repo(tmp)
            agent = RepositoryAnalysisAgent()
            meta  = _fake_meta("https://github.com/user/repo", "user", "repo")
            a1_result = agent.analyse_single(
                owner="user", name="repo",
                local_path=tmp, metadata=meta,
            )
            a1_summary = {
                "metadata":    meta,
                "statistics":  a1_result["statistics"],
                "knowledge":   a1_result["knowledge"],
                "report_path": a1_result["report_path"],
            }
            # Simulate the _pipeline_result["per_repo"] entry
            per_repo_entry = {"agent1": a1_summary, "agent2": {}}
            # Dashboard fallback: data = _pipeline_result["per_repo"][full_name]
            data = per_repo_entry
            assert data["agent1"]["statistics"]["total_files"] >= 0
            assert "files" in data["agent1"]["knowledge"]
