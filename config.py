"""
CodeLens AI — Central Configuration
All tuneable parameters live here. Override via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REPO_DIR        = os.path.join(BASE_DIR, "repositories")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
GRAPH_DIR       = os.path.join(BASE_DIR, "graphs")
DATA_DIR        = os.path.join(BASE_DIR, "data")

for _d in (REPO_DIR, OUTPUT_DIR, GRAPH_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────
# Input constraints
# ──────────────────────────────────────────────
MIN_REPOS = 1   # minimum enforced by validation; UI still recommends 5–10
MAX_REPOS = 10

# ──────────────────────────────────────────────
# Relevance scoring weights (must sum ≤ 1 each)
# ──────────────────────────────────────────────
SCORING_WEIGHTS = {
    "name":        0.25,
    "description": 0.20,
    "topics":      0.20,
    "readme":      0.15,
    "code":        0.12,
    "language":    0.08,
}

# ──────────────────────────────────────────────
# Analysis settings
# ──────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 1_000_000          # 1 MB — skip larger files
IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    ".tox", "dist", "build", ".eggs", "*.egg-info",
    ".mypy_cache", ".pytest_cache", "vendor", "third_party",
}
SUPPORTED_LANGUAGES = ["Python", "Java"]
CLONE_TIMEOUT_SECONDS = 120
MAX_README_CHARS = 8000                   # chars to use from README for scoring

# ──────────────────────────────────────────────
# GitHub API
# ──────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"

# ──────────────────────────────────────────────
# LLM / RAG
# ──────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "openai")     # openai | groq | ollama
LLM_API_KEY    = os.getenv("LLM_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RETRIEVAL = 5

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY", "codelens-dev-secret-change-in-prod")
DEBUG        = os.getenv("FLASK_DEBUG", "true").lower() == "true"
HOST         = os.getenv("FLASK_HOST", "0.0.0.0")
PORT         = int(os.getenv("FLASK_PORT", "5000"))
