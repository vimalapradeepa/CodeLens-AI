"""
Repository Cloner
Clones a GitHub repository into the local repositories/ directory.

Design decisions for large-repo robustness:
- Uses subprocess.Popen (not subprocess.run) to avoid the Windows pipe-buffer
  deadlock that occurs when git writes hundreds of MB of progress output to
  stderr and capture_output=True buffers it all in memory.
- Redirects git's verbose progress output to /dev/null (DEVNULL) so the pipe
  never fills up.  Actual error messages are captured via a separate stderr pipe
  limited to a small ring buffer.
- Shallow clone (--depth 1) keeps disk usage and transfer time minimal.
- --no-tags further reduces transfer size.
- --single-branch limits to the default branch only.
- Configurable timeout with a generous default for large repositories.
- On timeout the partial clone directory is cleaned up automatically.
"""

import os
import shutil
import logging
import subprocess
import threading
import time
from typing import Dict

from config import REPO_DIR, CLONE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Extra git environment: disable interactive prompts and progress meters
_GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",        # never prompt for credentials
    "GIT_ASKPASS":         "echo",     # return empty string for any ask
    "GIT_SSH_COMMAND":     "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
}

# How many characters to keep from the tail of stderr for error messages
_STDERR_TAIL_CHARS = 2000

# Timeout for large repos — use a larger value than the config default
# so that repos like apache/spark (800 MB) have time to clone.
_LARGE_REPO_TIMEOUT = max(CLONE_TIMEOUT_SECONDS, 600)   # at least 10 minutes


def _repo_local_name(owner: str, name: str) -> str:
    return f"{owner}__{name}"


def _drain_stderr(proc, buf: list):
    """
    Read stderr from `proc` in a background thread, keeping only the last
    _STDERR_TAIL_CHARS bytes.  This prevents the stderr pipe from blocking
    when git writes large amounts of output.
    """
    try:
        for line in iter(proc.stderr.readline, b""):
            buf.append(line)
            # Trim buffer to avoid unbounded memory growth
            total = sum(len(l) for l in buf)
            while total > _STDERR_TAIL_CHARS * 2 and len(buf) > 1:
                removed = buf.pop(0)
                total -= len(removed)
    except Exception:
        pass


def clone_repository(owner: str, name: str, clone_url: str) -> Dict:
    """
    Clone or reuse an already-cloned repository.

    Uses Popen + a stderr-draining thread to avoid the Windows pipe-buffer
    deadlock that kills large-repo clones mid-checkout.

    Returns:
        {
            "success":        bool,
            "local_path":     str | None,
            "message":        str,
            "already_existed":bool,
            "stderr_tail":    str,   # last few lines of git output (for debugging)
        }
    """
    local_name = _repo_local_name(owner, name)
    local_path = os.path.join(REPO_DIR, local_name)

    # ── Reuse already-cloned repo ─────────────────────────────────────────
    if os.path.isdir(local_path):
        logger.info("Repository already cloned, reusing: %s", local_path)
        return {
            "success":         True,
            "local_path":      local_path,
            "message":         "Repository already cloned (reused).",
            "already_existed": True,
            "stderr_tail":     "",
        }

    # ── Build clone URL ───────────────────────────────────────────────────
    safe_url = clone_url if clone_url else f"https://github.com/{owner}/{name}.git"

    # ── Build git command ─────────────────────────────────────────────────
    cmd = [
        "git", "clone",
        "--depth",         "1",      # shallow: only latest snapshot
        "--single-branch",           # don't fetch other branches
        "--no-tags",                 # skip tags to reduce transfer size
        "--progress",                # emit progress to stderr (we drain it)
        safe_url,
        local_path,
    ]

    logger.info("Cloning %s → %s (timeout=%ss)", safe_url, local_path, _LARGE_REPO_TIMEOUT)

    stderr_buf: list = []
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,   # we don't need stdout
            stderr=subprocess.PIPE,      # capture stderr for error messages
            env=_GIT_ENV,
        )

        # Drain stderr in background thread to prevent pipe deadlock
        drain_thread = threading.Thread(
            target=_drain_stderr, args=(proc, stderr_buf), daemon=True
        )
        drain_thread.start()

        # Wait for the process with timeout
        try:
            proc.wait(timeout=_LARGE_REPO_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            drain_thread.join(timeout=2)
            stderr_tail = _buf_to_str(stderr_buf)
            msg = (
                f"Clone timed out after {_LARGE_REPO_TIMEOUT}s. "
                f"The repository may be too large or the connection too slow. "
                f"Try again or check your internet connection."
            )
            logger.error("%s | stderr tail: %s", msg, stderr_tail[-200:])
            _cleanup(local_path)
            return {
                "success":         False,
                "local_path":      None,
                "message":         msg,
                "already_existed": False,
                "stderr_tail":     stderr_tail,
            }

        drain_thread.join(timeout=2)
        stderr_tail = _buf_to_str(stderr_buf)
        returncode = proc.returncode

        if returncode != 0:
            # Extract the meaningful part of the error (skip progress lines)
            clean_err = _extract_error(stderr_tail)
            msg = f"Clone failed (exit {returncode}): {clean_err}"
            logger.error(msg)
            _cleanup(local_path)
            return {
                "success":         False,
                "local_path":      None,
                "message":         msg,
                "already_existed": False,
                "stderr_tail":     stderr_tail,
            }

        logger.info("Clone successful: %s", local_path)
        return {
            "success":         True,
            "local_path":      local_path,
            "message":         "Cloned successfully.",
            "already_existed": False,
            "stderr_tail":     "",
        }

    except FileNotFoundError:
        msg = "git not found. Please install git and add it to PATH."
        logger.error(msg)
        return {
            "success":         False,
            "local_path":      None,
            "message":         msg,
            "already_existed": False,
            "stderr_tail":     "",
        }
    except Exception as exc:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        msg = f"Unexpected clone error: {exc}"
        logger.exception(msg)
        _cleanup(local_path)
        return {
            "success":         False,
            "local_path":      None,
            "message":         msg,
            "already_existed": False,
            "stderr_tail":     _buf_to_str(stderr_buf),
        }


def delete_repository(owner: str, name: str) -> bool:
    """Remove a cloned repository to free disk space."""
    local_path = os.path.join(REPO_DIR, _repo_local_name(owner, name))
    if os.path.isdir(local_path):
        shutil.rmtree(local_path, ignore_errors=True)
        return True
    return False


# ── Helpers ───────────────────────────────────────────────────────────────

def _cleanup(path: str):
    """Remove a partial clone directory silently."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _buf_to_str(buf: list) -> str:
    """Decode the collected stderr bytes into a string."""
    raw = b"".join(buf)
    return raw.decode("utf-8", errors="replace")[-_STDERR_TAIL_CHARS:]


def _extract_error(stderr_text: str) -> str:
    """
    Filter git stderr to remove progress lines (lines containing \\r or %)
    and return only the meaningful error lines.
    """
    lines = stderr_text.splitlines()
    error_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and "\r" not in line
        and "%" not in line
        and not line.startswith("remote: Counting")
        and not line.startswith("remote: Compressing")
        and not line.startswith("Receiving objects")
        and not line.startswith("Resolving deltas")
        and not line.startswith("Updating files")
        and not line.startswith("Checking out files")
    ]
    return " | ".join(error_lines[-10:]) if error_lines else stderr_text[-300:].strip()
