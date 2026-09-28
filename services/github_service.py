"""
GitHub Service
Fetches repository metadata from the GitHub REST API.
Handles rate-limiting gracefully and works without a token (60 req/hr).

Language detection is based solely on the actual source-code byte counts returned
by the GitHub /languages API endpoint.  Keywords, topics, README text, and
repository name are never used to determine primary_language.

Rate-limit budget (unauthenticated = 60 req/hr):
  Discovery uses 2 calls per repo  (main + languages).
  Topics come from the main repo response for free.
  README is fetched only when explicitly needed (not during discovery).
  An in-process cache prevents re-fetching the same repo in the same session.
"""

import requests
import time
import logging
from typing import Dict, Optional, List

from config import GITHUB_TOKEN, GITHUB_API_BASE, MAX_README_CHARS
from services.url_validator import parse_owner_repo

logger = logging.getLogger(__name__)

_HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# ── Session-level metadata cache ─────────────────────────────────────────
# Keyed by normalised GitHub URL (no trailing slash, no .git).
# Cleared on server restart; avoids re-fetching during the same session.
_meta_cache: Dict[str, Dict] = {}


def _cache_key(url: str) -> str:
    return url.rstrip("/").replace(".git", "").lower()

# ── Language classification ───────────────────────────────────────────────
#
# GitHub's /languages endpoint counts bytes per language found in actual source
# files.  However, GitHub sometimes reports a non-code format (notebook, markup,
# config, data) as "primary" when it dominates by raw byte count.  We correct
# this by skipping non-code entries and picking the dominant true code language.
#
# NON_CODE_LANGUAGES: formats that are data/markup/config, not programming langs.
# Sorted so the set lookup is O(1); the actual order doesn't matter.

_NON_CODE_LANGUAGES = {
    # Notebooks
    "jupyter notebook",
    # Markup & docs
    "html", "css", "scss", "sass", "less", "markdown", "restructuredtext",
    "asciidoc", "tex", "roff", "pod", "rdoc", "wiki",
    # Config / data
    "yaml", "json", "toml", "ini", "xml", "hcl", "dockerfile",
    "makefile", "cmake", "batchfile", "nix", "m4",
    # Template languages (not standalone programming languages)
    "smarty", "mustache", "handlebars", "jinja", "liquid", "twig",
    "erb", "ejs",
    # Shell / scripting that isn't a "language" per se
    "shell", "powershell", "awk", "sed",
    # Other non-code
    "svg", "rich text format",
}

# True programming languages in preference order.
# When multiple code languages are present the one highest on this list wins.
# The list is intentionally broad; languages not on it are still valid — they
# are returned in GitHub byte-count order (highest first) as a final fallback.
_CODE_LANGUAGE_PRIORITY: List[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#",
    "Go", "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R",
    "MATLAB", "Perl", "Haskell", "Lua", "Dart", "Elixir", "Clojure",
    "Groovy", "Julia", "Fortran", "COBOL", "Assembly", "Objective-C",
    "F#", "Erlang", "Nim", "Crystal", "Zig", "V",
    # SQL dialects are real code
    "PLSQL", "PLpgSQL", "TSQL", "SQL",
    # Less common but legitimate
    "CoffeeScript", "LiveScript", "Elm", "PureScript", "ReasonML",
    "OCaml", "Standard ML", "Racket", "Scheme", "Common Lisp",
    "Prolog", "Pascal", "Delphi", "Ada", "VHDL", "Verilog",
]

# Build a lowercase → canonical mapping from the priority list for fast lookup.
_PRIORITY_LOWER: Dict[str, str] = {lang.lower(): lang for lang in _CODE_LANGUAGE_PRIORITY}


def _best_language(github_primary: Optional[str], langs_data: Dict) -> Optional[str]:
    """
    Return the dominant **programming** language for a repository, determined
    entirely from the actual source-code bytes reported by GitHub's /languages
    endpoint.

    Decision order:
    1. Scan langs_data (already sorted by byte count, highest first) for the
       first entry that is a real code language (not in _NON_CODE_LANGUAGES).
       This is the authoritative answer when the breakdown is available.
    2. If langs_data is empty (API call failed / rate-limited), fall back to
       data.get("language") from the main repo endpoint — but still skip
       non-code labels using the same filter.
    3. If nothing is available → return None (template renders "—").

    Note: keywords, topics, README, and repository name play NO part here.
    """
    # ── Case 1: use the byte-count breakdown (most accurate) ────────────
    if langs_data:
        # First pass: check priority list to prefer well-known languages
        for preferred in _CODE_LANGUAGE_PRIORITY:
            if preferred in langs_data:
                return preferred

        # Second pass: any entry not in the non-code set (GitHub byte-count order)
        for lang in langs_data:
            if lang.lower() not in _NON_CODE_LANGUAGES:
                return lang

        # All entries are non-code (pure notebook/markup repo) — return the
        # highest-byte-count entry as a best-effort label.
        return next(iter(langs_data))

    # ── Case 2: breakdown unavailable, use main-repo language field ──────
    if github_primary:
        if github_primary.lower() not in _NON_CODE_LANGUAGES:
            # github_primary is a real code language — use it
            return github_primary
        # github_primary is a non-code label (e.g. "HTML", "Jupyter Notebook")
        # and langs_data is empty (rate-limited).  We cannot determine the real
        # code language, so return None rather than displaying a misleading
        # non-code label in the Language column.
        return None

    # ── Case 3: no data at all ────────────────────────────────────────────
    return None


def _all_code_languages(langs_data: Dict, github_primary: Optional[str] = None) -> List[str]:
    """
    Return all actual programming languages present in the repository,
    ordered by byte count (highest first), with non-code entries removed.

    Used for the Language column in Discovery so every language is shown,
    not just the single dominant one.

    Falls back to [primary_language] when langs_data is unavailable,
    or [] when nothing meaningful can be determined.
    """
    if langs_data:
        result = [
            lang for lang in langs_data          # already byte-count-sorted by GitHub
            if lang.lower() not in _NON_CODE_LANGUAGES
        ]
        return result if result else []

    # langs_data unavailable (rate-limited) — use github_primary as a single entry
    if github_primary and github_primary.lower() not in _NON_CODE_LANGUAGES:
        return [github_primary]

    return []


def _get(path: str, params: Optional[Dict] = None,
         timeout: int = 8) -> Optional[Dict]:
    """
    Single GitHub API GET.

    - Hard request timeout: `timeout` seconds (default 8).
    - On rate-limit (403): waits at most 5 s then gives up (avoids blocking
      the discovery pipeline for the full reset window).
    - On network error: one retry only, no exponential back-off.
    - Never retries on 404 or other 4xx.
    """
    url = f"{GITHUB_API_BASE}{path}"
    for attempt in range(2):          # max 2 attempts (1 retry)
        try:
            resp = requests.get(url, headers=_HEADERS, params=params,
                                timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                # Rate-limited — wait briefly then abort this call
                reset = int(resp.headers.get("X-RateLimit-Reset",
                                             time.time() + 5))
                wait = max(1, min(reset - int(time.time()), 5))  # cap at 5 s
                logger.warning("Rate-limited on %s. Waiting %ss …", path, wait)
                time.sleep(wait)
                return None           # don't retry after rate-limit
            if resp.status_code == 404:
                logger.debug("Not found: %s", url)
                return None
            logger.error("GitHub API %s → HTTP %s", url, resp.status_code)
            return None
        except requests.Timeout:
            logger.warning("Timeout on %s (attempt %d)", url, attempt + 1)
            # One retry on timeout, then give up
            if attempt == 0:
                continue
            return None
        except requests.RequestException as exc:
            logger.error("Request error on %s: %s", url, exc)
            return None
    return None


def fetch_repo_metadata(github_url: str, include_readme: bool = False) -> Dict:
    """
    Return a metadata dict for the given GitHub URL.

    Uses at most 2 API calls (main repo + languages breakdown).
    Topics are extracted from the main repo response for free.
    README is only fetched when include_readme=True (skipped during discovery
    to conserve rate-limit quota).

    Results are cached in _meta_cache for the lifetime of the process so that
    repeated calls for the same URL (e.g. multiple "Find & Analyze" clicks in
    the same session) do not consume additional API quota.

    On failure, returns a dict with 'error' key set and all display fields
    populated with safe defaults so the Discovery table never shows blank rows.
    """
    ckey = _cache_key(github_url)
    if ckey in _meta_cache:
        logger.debug("Cache hit for %s", github_url)
        return _meta_cache[ckey]

    try:
        owner, repo = parse_owner_repo(github_url)
    except ValueError as exc:
        return _error_meta(github_url, str(exc))

    # ── Call 1: main repo info ────────────────────────────────────────────
    data = _get(f"/repos/{owner}/{repo}")
    if data is None:
        meta = _error_meta(github_url,
                           "Repository not found or inaccessible (may be private or rate-limited).")
        meta["owner"] = owner
        meta["name"]  = repo
        meta["full_name"] = f"{owner}/{repo}"
        meta["url"] = github_url
        _meta_cache[ckey] = meta
        return meta

    # Topics come from the main repo response — no extra API call needed.
    topics = data.get("topics") or []

    # ── Call 2: languages breakdown (for accurate primary_language) ───────
    langs_data = _get(f"/repos/{owner}/{repo}/languages") or {}
    total_bytes = sum(langs_data.values()) or 1
    languages = {k: round(v / total_bytes * 100, 1) for k, v in langs_data.items()}

    # If languages call was rate-limited, fall back to the main response field.
    primary_language = _best_language(data.get("language"), langs_data)

    # All actual code languages present in the repo (non-code entries removed),
    # ordered by byte count (highest first).  Used for the Language column in
    # Discovery so users see every language, not just the dominant one.
    code_languages = _all_code_languages(langs_data, data.get("language"))

    # ── Call 3 (optional): README ─────────────────────────────────────────
    readme_text = ""
    if include_readme:
        readme_data = _get(f"/repos/{owner}/{repo}/readme")
        if readme_data:
            import base64
            try:
                readme_text = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="ignore")
                readme_text = readme_text[:MAX_README_CHARS]
            except Exception:
                readme_text = ""

    meta = {
        "url":              github_url,
        "owner":            owner,
        "name":             repo,
        "full_name":        data.get("full_name", f"{owner}/{repo}"),
        "description":      data.get("description") or "",
        "topics":           topics,
        "languages":        languages,
        "primary_language": primary_language,
        "code_languages":   code_languages,
        "stars":            data.get("stargazers_count", 0),
        "forks":            data.get("forks_count", 0),
        "open_issues":      data.get("open_issues_count", 0),
        "size_kb":          data.get("size", 0),
        "default_branch":   data.get("default_branch", "main"),
        "readme":           readme_text,
        "clone_url":        data.get("clone_url", ""),
        "created_at":       data.get("created_at", ""),
        "updated_at":       data.get("updated_at", ""),
        "error":            None,
    }
    _meta_cache[ckey] = meta
    return meta


def fetch_repos_metadata(urls: list, max_workers: int = 5,
                         per_repo_timeout: float = 10.0) -> list:
    """
    Fetch metadata for a list of URLs in parallel.

    - Each URL is processed in its own thread (up to `max_workers` at once).
    - Each repo has a hard wall-clock deadline of `per_repo_timeout` seconds
      enforced by future.result(timeout=...) — no individual repo can stall
      the others beyond that limit.
    - If a repo times out or raises, it gets an error dict; the rest continue.
    - Result order matches the input URL order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not urls:
        return []

    ordered_results: Dict[int, Dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(fetch_repo_metadata, url): (idx, url)
            for idx, url in enumerate(urls)
        }

        # Collect futures as they complete; each gets its own deadline.
        for future in as_completed(future_to_idx):
            idx, url = future_to_idx[future]
            try:
                meta = future.result(timeout=per_repo_timeout)
            except TimeoutError:
                logger.warning("Metadata fetch timed out for %s", url)
                meta = _timeout_meta(url)
            except Exception as exc:
                logger.warning("Metadata fetch failed for %s: %s", url, exc)
                meta = _error_meta(url, str(exc))

            if meta.get("error"):
                logger.warning("Unavailable %s: %s", url, meta["error"])
            ordered_results[idx] = meta

    # Safety net: fill any slot that somehow never produced a result.
    for idx, url in enumerate(urls):
        if idx not in ordered_results:
            logger.warning("No result for %s — marking as unavailable", url)
            ordered_results[idx] = _timeout_meta(url)

    return [ordered_results[i] for i in range(len(urls))]


def _timeout_meta(url: str) -> Dict:
    """Fallback metadata dict when a fetch times out."""
    try:
        owner, name = parse_owner_repo(url)
    except Exception:
        owner, name = "", url
    return {
        "url": url, "owner": owner, "name": name,
        "full_name": f"{owner}/{name}" if owner else url,
        "error": "Request timed out — repository may be slow or unavailable.",
        "description": "", "topics": [], "languages": {},
        "primary_language": None, "code_languages": [], "stars": 0, "forks": 0,
        "open_issues": 0, "size_kb": 0, "default_branch": "main",
        "readme": "", "clone_url": "", "created_at": "", "updated_at": "",
    }


def _error_meta(url: str, message: str) -> Dict:
    """Fallback metadata dict when a fetch raises an unexpected exception."""
    try:
        owner, name = parse_owner_repo(url)
    except Exception:
        owner, name = "", url
    return {
        "url": url, "owner": owner, "name": name,
        "full_name": f"{owner}/{name}" if owner else url,
        "error": f"Metadata fetch failed: {message}",
        "description": "", "topics": [], "languages": {},
        "primary_language": None, "code_languages": [], "stars": 0, "forks": 0,
        "open_issues": 0, "size_kb": 0, "default_branch": "main",
        "readme": "", "clone_url": "", "created_at": "", "updated_at": "",
    }


def clear_metadata_cache():
    """Clear the in-process metadata cache (useful for testing)."""
    global _meta_cache
    _meta_cache = {}
