"""
Tests for the per-repository analysis state machine in app.py.

Key behaviours verified:
  - Each repository has its own independent status in _repo_status.
  - Two different repositories can be analysed simultaneously.
  - The same repository is blocked (409) while its own analysis is running.
  - A different repository is NEVER blocked by another repo's analysis.
  - running=False after success, clone failure, and uncaught exception.
  - done=True only after a successful analysis (not from discovery data).
  - /api/status?repo=X returns only X's status.
  - /api/analysis/reset clears the running flag for a specific repo.
  - Stale lock (>MAX seconds) is auto-cleared so the same repo can re-analyze.
"""

import json
import threading
import time
import pytest
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────

REPO_A = "apache/airflow"
REPO_B = "apache/spark"


def _fresh(full_name="", **overrides):
    """Return a fresh status dict matching _fresh_repo_status()."""
    base = {
        "running":          False,
        "steps":            [],
        "error":            None,
        "analysis_started": False,
        "clone_done":       False,
        "agent1_done":      False,
        "agent2_done":      False,
        "rag_done":         False,
        "started_at":       None,
        "current_repo":     full_name,
    }
    base.update(overrides)
    return base


def _fake_meta(full_name="user/repo", **kw):
    base = {
        "name": full_name.split("/")[-1],
        "full_name": full_name,
        "owner": full_name.split("/")[0],
        "url": f"https://github.com/{full_name}",
        "description": "", "topics": [],
        "primary_language": "Python", "languages": {},
        "code_languages": ["Python"],
        "stars": 0, "forks": 0, "open_issues": 0, "size_kb": 0,
        "default_branch": "main", "readme": "", "clone_url": "",
        "created_at": "", "updated_at": "", "error": None,
        "code_entities_text": "",
    }
    base.update(kw)
    return base


def _fake_clone(success=True, msg="ok"):
    def _fn(*a, **k):
        return {"success": success, "local_path": "/tmp/fake",
                "message": msg, "already_existed": True, "stderr_tail": ""}
    return _fn


# ── App fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_app_state():
    import app as flask_app
    flask_app._repo_status     = {}
    flask_app._pipeline_result = {}
    flask_app._orchestrator    = None
    flask_app._qa_engine       = None
    yield
    flask_app._repo_status     = {}
    flask_app._pipeline_result = {}


@pytest.fixture
def client():
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────

def _status(client, repo=""):
    url = f"/api/status?repo={repo}" if repo else "/api/status"
    return json.loads(client.get(url).data)


def _analyze(client, full_name="user/repo"):
    return client.post(
        "/api/analyze",
        data=json.dumps({"full_name": full_name}),
        content_type="application/json",
    )


def _wait_done(client, repo, timeout=3.0):
    """Wait until the repo's analysis is no longer running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _status(client, repo)["running"]:
            return
        time.sleep(0.05)


# ── Initial state ─────────────────────────────────────────────────────────

class TestInitialState:

    def test_no_status_initially(self, client):
        s = _status(client, REPO_A)
        assert s["running"] is False
        assert s["error"]   is None

    def test_done_false_initially(self, client):
        assert _status(client, REPO_A)["done"] is False

    def test_stage_flags_false_initially(self, client):
        s = _status(client, REPO_A)
        for flag in ("clone_done", "agent1_done", "agent2_done", "rag_done"):
            assert s[flag] is False


# ── Independent concurrent analyses ──────────────────────────────────────

class TestIndependentAnalyses:

    def test_two_different_repos_both_accepted(self, client):
        """
        CORE REQUIREMENT: starting Spark while Airflow is running
        must NOT return 409.
        """
        import app as flask_app

        # Put REPO_A in running state manually
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True,
            started_at=time.monotonic()
        )

        # REPO_B should start without any error
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_B)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp = _analyze(client, full_name=REPO_B)

        assert resp.status_code == 200, (
            f"Expected 200 for {REPO_B} while {REPO_A} is running, "
            f"got {resp.status_code}: {resp.data.decode()}"
        )
        data = json.loads(resp.data)
        assert data.get("started") is True

    def test_repo_a_status_unaffected_by_repo_b(self, client):
        """Starting Repo B must not change Repo A's status."""
        import app as flask_app

        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True,
            clone_done=True, started_at=time.monotonic()
        )

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_B)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            _analyze(client, full_name=REPO_B)

        _wait_done(client, REPO_B)

        sa = _status(client, REPO_A)
        assert sa["running"]    is True
        assert sa["clone_done"] is True

    def test_status_is_per_repo(self, client):
        """Each repo's status is independent."""
        import app as flask_app

        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, clone_done=True, agent1_done=True
        )
        flask_app._repo_status[REPO_B] = _fresh(
            REPO_B, running=False, clone_done=False
        )

        sa = _status(client, REPO_A)
        sb = _status(client, REPO_B)

        assert sa["running"]    is True
        assert sa["clone_done"] is True
        assert sa["agent1_done"]is True
        assert sb["running"]    is False
        assert sb["clone_done"] is False


# ── Same-repo protection ──────────────────────────────────────────────────

class TestSameRepoProtection:

    def test_same_repo_409_while_running(self, client):
        """A second Analyze for the SAME repo must return 409."""
        import app as flask_app
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True,
            started_at=time.monotonic()
        )

        resp = _analyze(client, full_name=REPO_A)
        assert resp.status_code == 409
        data = json.loads(resp.data)
        assert "running" in data.get("error", "").lower()
        assert data.get("current_repo") == REPO_A

    def test_same_repo_accepted_after_it_finishes(self, client):
        """After the first analysis finishes, the same repo can be re-analyzed."""
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_A)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            _analyze(client, full_name=REPO_A)

        _wait_done(client, REPO_A)
        assert _status(client, REPO_A)["running"] is False

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_A)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp2 = _analyze(client, full_name=REPO_A)

        assert resp2.status_code == 200
        assert json.loads(resp2.data).get("error") != "Analysis already running for this repository."

    def test_stale_same_repo_lock_auto_cleared(self, client):
        import app as flask_app
        old_start = time.monotonic() - (flask_app._MAX_ANALYSIS_SECONDS + 10)
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True,
            started_at=old_start
        )

        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_A)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp = _analyze(client, full_name=REPO_A)

        assert resp.status_code == 200


# ── State lifecycle ───────────────────────────────────────────────────────

class TestStateLifecycle:

    def _run_with_clone_failure(self, client, repo=REPO_A, msg="Clone failed"):
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(repo)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg=msg)):
            _analyze(client, full_name=repo)
        _wait_done(client, repo)
        return _status(client, repo)

    def test_running_cleared_after_clone_failure(self, client):
        s = self._run_with_clone_failure(client)
        assert s["running"] is False

    def test_error_set_after_clone_failure(self, client):
        s = self._run_with_clone_failure(client, msg="Clone failed: network error")
        assert s["error"] is not None

    def test_done_false_after_clone_failure(self, client):
        assert self._run_with_clone_failure(client)["done"] is False

    def test_running_cleared_after_exception(self, client):
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=RuntimeError("crash")):
            _analyze(client, full_name=REPO_A)
        _wait_done(client, REPO_A)
        s = _status(client, REPO_A)
        assert s["running"] is False
        assert s["error"]   is not None

    def test_second_repo_accepted_after_first_repo_fails(self, client):
        """After REPO_A fails, REPO_B can still start (it's a different repo)."""
        self._run_with_clone_failure(client, repo=REPO_A)
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_B)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp = _analyze(client, full_name=REPO_B)
        assert resp.status_code == 200


# ── Reset endpoint ────────────────────────────────────────────────────────

class TestResetEndpoint:

    def test_reset_specific_repo(self, client):
        import app as flask_app
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True
        )

        resp = client.post(
            "/api/analysis/reset",
            data=json.dumps({"full_name": REPO_A}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert json.loads(resp.data)["reset"] is True
        assert _status(client, REPO_A)["running"] is False

    def test_reset_does_not_affect_other_repo(self, client):
        import app as flask_app
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True
        )
        flask_app._repo_status[REPO_B] = _fresh(
            REPO_B, running=True, analysis_started=True
        )

        client.post(
            "/api/analysis/reset",
            data=json.dumps({"full_name": REPO_A}),
            content_type="application/json",
        )

        assert _status(client, REPO_A)["running"] is False
        assert _status(client, REPO_B)["running"] is True

    def test_reset_when_not_running_returns_false(self, client):
        resp = client.post(
            "/api/analysis/reset",
            data=json.dumps({"full_name": REPO_A}),
            content_type="application/json",
        )
        assert json.loads(resp.data)["reset"] is False

    def test_reset_allows_same_repo_to_restart(self, client):
        import app as flask_app
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True
        )
        client.post(
            "/api/analysis/reset",
            data=json.dumps({"full_name": REPO_A}),
            content_type="application/json",
        )
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_A)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp = _analyze(client, full_name=REPO_A)
        assert resp.status_code == 200


# ── Full lifecycle ────────────────────────────────────────────────────────

class TestFullLifecycle:

    def test_airflow_and_spark_simultaneously(self, client):
        """
        Simulate: Airflow is running → click Analyze for Spark →
        Spark must start (200), Airflow status is unaffected.
        """
        import app as flask_app

        # Airflow is already running
        flask_app._repo_status[REPO_A] = _fresh(
            REPO_A, running=True, analysis_started=True,
            clone_done=True, started_at=time.monotonic()
        )

        # Start Spark
        with patch("services.github_service.fetch_repo_metadata",
                   side_effect=lambda *a, **k: _fake_meta(REPO_B)), \
             patch("services.repository_cloner.clone_repository",
                   side_effect=_fake_clone(success=False, msg="fail")):
            resp = _analyze(client, full_name=REPO_B)

        assert resp.status_code == 200, "Spark must start while Airflow is running"

        # Wait for Spark to finish (clone failure is fast)
        _wait_done(client, REPO_B)

        # Airflow's status must be unchanged
        sa = _status(client, REPO_A)
        assert sa["running"]    is True
        assert sa["clone_done"] is True

        # Spark's status reflects its own failure
        sb = _status(client, REPO_B)
        assert sb["running"] is False
        assert sb["error"]   is not None
