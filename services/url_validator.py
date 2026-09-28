"""
URL Validation Service
Validates GitHub repository URLs, checks for duplicates, and enforces limits.
"""

import re
from typing import List, Tuple, Dict

# Matches:  https://github.com/owner/repo  (trailing slash and .git optional)
_GITHUB_PATTERN = re.compile(
    r'^https?://(?:www\.)?github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?/?$'
)


def validate_github_url(url: str) -> Tuple[bool, str]:
    """
    Return (is_valid, error_message).
    error_message is '' when valid.
    """
    url = url.strip()
    if not url:
        return False, "URL cannot be empty."
    if not url.startswith("http"):
        return False, "URL must start with http:// or https://."
    m = _GITHUB_PATTERN.match(url)
    if not m:
        return False, f"Not a valid GitHub repository URL: {url}"
    return True, ""


def normalize_url(url: str) -> str:
    """Strip trailing slash and .git suffix for consistent comparison."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def parse_owner_repo(url: str) -> Tuple[str, str]:
    """Extract (owner, repo) from a validated GitHub URL."""
    m = _GITHUB_PATTERN.match(url.strip())
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse owner/repo from URL: {url}")


def validate_url_list(urls: List[str]) -> Dict:
    """
    Validate a list of GitHub URLs.

    Returns a dict with:
        valid      : list of normalised valid unique URLs
        errors     : list of {url, message} dicts  — hard failures
        warnings   : list of {url, message} dicts  — soft notices (duplicates)
        duplicates : list of duplicate raw URLs that were removed
        count      : number of valid unique URLs
    """
    from config import MIN_REPOS, MAX_REPOS

    seen = set()
    valid = []
    errors = []
    warnings = []
    duplicates = []

    for raw in urls:
        url = raw.strip()
        if not url:
            continue
        ok, msg = validate_github_url(url)
        if not ok:
            errors.append({"url": url, "message": msg})
            continue
        norm = normalize_url(url)
        if norm in seen:
            duplicates.append(url)
            warnings.append({
                "url": url,
                "message": f"Duplicate URL ignored — already included: {url}",
            })
            continue
        seen.add(norm)
        valid.append(norm)

    if len(valid) < MIN_REPOS:
        errors.append({
            "url": "",
            "message": "Please submit at least one valid repository URL."
        })

    if len(valid) > MAX_REPOS:
        extras = valid[MAX_REPOS:]
        valid = valid[:MAX_REPOS]
        for e in extras:
            errors.append({"url": e, "message": f"Exceeded maximum of {MAX_REPOS} repositories — ignored."})

    return {
        "valid":      valid,
        "errors":     errors,
        "warnings":   warnings,
        "duplicates": duplicates,
        "count":      len(valid),
    }
