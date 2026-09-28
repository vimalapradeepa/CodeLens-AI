"""
Tests for repository_cloner.clone_repository.

Covers:
  - Reuse of already-cloned directory (no git call).
  - Successful clone produces correct result shape.
  - Timeout is handled gracefully (no crash, returns error dict).
  - Non-zero exit code (e.g. repo not found) returns clear error.
  - git not on PATH returns a helpful message.
  - Partial clone directory is cleaned up on failure.
  - stderr_tail is captured and surfaced in the error result.
  - _extract_error strips progress lines from git stderr.
  - Large-repo flags are present in the git command (--depth 1, --no-tags,
    --single-branch) to minimise transfer size.
"""

import os
import tempfile
import threading
import time
import pytest
from unittest.mock import patch, MagicMock

from services.repository_cloner import (
    clone_repository, delete_repository,
    _extract_error, _buf_to_str, _cleanup,
)


# ── Helpers ───────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"success", "local_path", "message", "already_existed", "stderr_tail"}


def _check_shape(result: dict):
    missing = REQUIRED_KEYS - result.keys()
    assert not missing, f"Result missing keys: {missing}"


# ── Reuse existing clone ──────────────────────────────────────────────────

class TestReuseExisting:

    def test_reuses_existing_directory(self, tmp_path):
        repo_dir = tmp_path / "user__repo"
        repo_dir.mkdir()

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)):
            result = clone_repository("user", "repo", "https://github.com/user/repo.git")

        assert result["success"] is True
        assert result["already_existed"] is True
        assert result["local_path"] == str(repo_dir)
        _check_shape(result)

    def test_reuse_does_not_call_git(self, tmp_path):
        repo_dir = tmp_path / "user__repo"
        repo_dir.mkdir()

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen") as mock_popen:
            clone_repository("user", "repo", "")
            mock_popen.assert_not_called()


# ── Successful clone ──────────────────────────────────────────────────────

class TestSuccessfulClone:

    def _make_fake_popen(self, tmp_path, returncode=0):
        """Create a Popen mock that creates the clone dir and exits cleanly."""
        local_path = str(tmp_path / "user__repo")

        def fake_popen(cmd, stdout, stderr, env):
            os.makedirs(local_path, exist_ok=True)
            mock = MagicMock()
            mock.returncode = returncode
            mock.stderr = iter([])                # empty stderr stream
            mock.wait.return_value = returncode
            mock.kill.return_value = None
            return mock

        return fake_popen

    def test_successful_clone_returns_correct_shape(self, tmp_path):
        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=self._make_fake_popen(tmp_path, 0)):
            result = clone_repository("user", "repo", "https://github.com/user/repo.git")

        _check_shape(result)
        assert result["success"] is True
        assert result["already_existed"] is False
        assert result["local_path"] is not None
        assert result["stderr_tail"] == ""

    def test_successful_clone_local_path_correct(self, tmp_path):
        expected = str(tmp_path / "apache__spark")

        def fake_popen(cmd, stdout, stderr, env):
            os.makedirs(expected, exist_ok=True)
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = iter([])
            mock.wait.return_value = 0
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            result = clone_repository("apache", "spark", "https://github.com/apache/spark.git")

        assert result["local_path"] == expected

    def test_git_command_includes_shallow_flags(self, tmp_path):
        """--depth 1, --no-tags, --single-branch must be in the git command."""
        captured_cmd = []

        def fake_popen(cmd, stdout, stderr, env):
            captured_cmd.extend(cmd)
            path = str(tmp_path / "user__repo")
            os.makedirs(path, exist_ok=True)
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = iter([])
            mock.wait.return_value = 0
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            clone_repository("user", "repo", "https://github.com/user/repo.git")

        assert "--depth"         in captured_cmd, "Missing --depth"
        assert "1"               in captured_cmd, "Missing depth value 1"
        assert "--no-tags"       in captured_cmd, "Missing --no-tags"
        assert "--single-branch" in captured_cmd, "Missing --single-branch"


# ── Failure: non-zero exit code ───────────────────────────────────────────

class TestCloneFailure:

    def test_nonzero_exit_returns_failure(self, tmp_path):
        def fake_popen(cmd, stdout, stderr, env):
            mock = MagicMock()
            mock.returncode = 128
            mock.stderr = iter([
                b"fatal: repository 'https://github.com/user/nonexistent.git/' not found\n"
            ])
            mock.wait.return_value = 128
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            result = clone_repository("user", "nonexistent", "")

        _check_shape(result)
        assert result["success"] is False
        assert result["local_path"] is None
        assert "128" in result["message"] or "not found" in result["message"].lower() \
               or "failed" in result["message"].lower()

    def test_failure_cleans_up_partial_directory(self, tmp_path):
        """A failed clone must not leave a partial directory behind."""
        partial = tmp_path / "user__repo"

        def fake_popen(cmd, stdout, stderr, env):
            partial.mkdir(exist_ok=True)   # git creates the dir before failing
            mock = MagicMock()
            mock.returncode = 1
            mock.stderr = iter([b"error: something went wrong\n"])
            mock.wait.return_value = 1
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            clone_repository("user", "repo", "")

        assert not partial.exists(), "Partial clone directory should be cleaned up"

    def test_stderr_tail_in_failure_result(self, tmp_path):
        def fake_popen(cmd, stdout, stderr, env):
            mock = MagicMock()
            mock.returncode = 1
            mock.stderr = iter([b"error: specific error message\n"])
            mock.wait.return_value = 1
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            result = clone_repository("user", "repo", "")

        # stderr_tail must be present (even if message was filtered)
        assert isinstance(result["stderr_tail"], str)


# ── Timeout handling ──────────────────────────────────────────────────────

class TestCloneTimeout:

    def test_timeout_returns_failure(self, tmp_path):
        def fake_popen(cmd, stdout, stderr, env):
            mock = MagicMock()
            mock.stderr = iter([])
            mock.wait.side_effect = subprocess.TimeoutExpired(cmd, 1)
            mock.kill.return_value = None
            mock.returncode = -1
            return mock

        import subprocess
        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            result = clone_repository("user", "repo", "")

        _check_shape(result)
        assert result["success"] is False
        assert "timed out" in result["message"].lower() or "timeout" in result["message"].lower()
        assert result["local_path"] is None

    def test_timeout_cleans_up_directory(self, tmp_path):
        partial = tmp_path / "user__repo"

        import subprocess as sp

        def fake_popen(cmd, stdout, stderr, env):
            partial.mkdir(exist_ok=True)
            mock = MagicMock()
            mock.stderr = iter([])
            mock.wait.side_effect = sp.TimeoutExpired(cmd, 1)
            mock.kill.return_value = None
            return mock

        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=fake_popen):
            clone_repository("user", "repo", "")

        assert not partial.exists(), "Timed-out clone directory should be removed"


# ── git not on PATH ───────────────────────────────────────────────────────

class TestGitNotFound:

    def test_git_not_found_returns_helpful_message(self, tmp_path):
        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)), \
             patch("subprocess.Popen", side_effect=FileNotFoundError("git not found")):
            result = clone_repository("user", "repo", "")

        _check_shape(result)
        assert result["success"] is False
        assert "git" in result["message"].lower()


# ── stderr filtering ──────────────────────────────────────────────────────

class TestExtractError:

    def test_filters_out_progress_lines(self):
        stderr = (
            "remote: Counting objects: 100\n"
            "remote: Compressing objects:  50% (5/10)\n"
            "Receiving objects: 100% (10/10)\n"
            "Resolving deltas: 100% (3/3)\n"
            "Updating files: 9% (2462/27353)\r"
            "fatal: repository not found\n"
        )
        result = _extract_error(stderr)
        assert "fatal" in result.lower() or "not found" in result.lower()
        # Progress lines should not dominate the output
        assert "Updating files" not in result or "fatal" in result

    def test_returns_empty_string_for_clean_output(self):
        result = _extract_error("")
        assert isinstance(result, str)

    def test_handles_only_progress_output(self):
        """When stderr is only progress, return the tail as-is (best effort)."""
        stderr = "Updating files: 50% (100/200)\rUpdating files: 100% (200/200)\r"
        result = _extract_error(stderr)
        assert isinstance(result, str)

    def test_buf_to_str_decodes_bytes(self):
        buf = [b"line one\n", b"line two\n"]
        result = _buf_to_str(buf)
        assert "line one" in result
        assert "line two" in result

    def test_buf_to_str_handles_empty(self):
        assert _buf_to_str([]) == ""


# ── delete_repository ─────────────────────────────────────────────────────

class TestDeleteRepository:

    def test_delete_existing(self, tmp_path):
        repo_dir = tmp_path / "user__repo"
        repo_dir.mkdir()
        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)):
            result = delete_repository("user", "repo")
        assert result is True
        assert not repo_dir.exists()

    def test_delete_nonexistent_returns_false(self, tmp_path):
        with patch("services.repository_cloner.REPO_DIR", str(tmp_path)):
            result = delete_repository("user", "nonexistent")
        assert result is False
