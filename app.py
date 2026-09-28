"""
CodeLens AI — Flask Application
Entry point for the web interface.
"""

import os
import json
import logging
import threading
from flask import (
    Flask, render_template, request, jsonify,
    send_file, session, redirect, url_for
)

from config import SECRET_KEY, DEBUG, HOST, PORT, OUTPUT_DIR, GRAPH_DIR
from agents.orchestrator import MultiAgentOrchestrator
from agents.repository_agent import RepositoryAnalysisAgent
from agents.architecture_agent import SemanticCodeGraphAgent
from services.url_validator import validate_url_list
from rag.qa_engine import QAEngine
from rag.retriever import VectorRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ── CORS: allow same-origin and localhost dev variants ────────────────────
@app.after_request
def _add_cors(response):
    """
    Allow the browser to call the API whether it's opened via
    http://127.0.0.1:5000, http://localhost:5000, or a LAN address.
    This is not a security risk because the app is a local dev tool.
    """
    origin = request.headers.get("Origin", "")
    if origin:
        response.headers["Access-Control-Allow-Origin"]  = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/health")
def api_health():
    """Simple health-check endpoint the frontend can ping to verify the server is up."""
    return jsonify({"status": "ok", "service": "CodeLens AI"})

# ── In-memory state (per-process; good enough for demo/prototype) ─────────
_orchestrator: MultiAgentOrchestrator = None
_pipeline_result: dict = {}

# Per-repository analysis status.
# Keyed by full_name (e.g. "apache/airflow").
# Replaces the old single _pipeline_status dict so multiple repositories
# can be analysed independently without blocking each other.
_repo_status: dict = {}   # {full_name: status_dict}

# Maximum seconds a single analysis is allowed to hold the running flag.
# After this the flag is treated as stale and auto-released.
_MAX_ANALYSIS_SECONDS = 30 * 60   # 30 minutes

_qa_engine: QAEngine = None
_lock = threading.Lock()


def _fresh_repo_status(full_name: str = "") -> dict:
    """Return a zeroed-out status dict for one repository."""
    return {
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


def _get_repo_status(full_name: str) -> dict:
    """Return the status dict for a repo (creates a fresh one if absent)."""
    if full_name not in _repo_status:
        _repo_status[full_name] = _fresh_repo_status(full_name)
    return _repo_status[full_name]


def _on_progress(step: str, msg: str, repo: str = ""):
    """Record a progress step for the given repo (or all active repos)."""
    with _lock:
        target = repo or ""
        if target and target in _repo_status:
            _repo_status[target]["steps"].append({"step": step, "msg": msg})
        elif not target:
            # Legacy: append to all running repos
            for st in _repo_status.values():
                if st.get("running"):
                    st["steps"].append({"step": step, "msg": msg})
    logger.info("Progress [%s]: %s — %s", repo, step, msg)


def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    with _lock:
        if _orchestrator is None:
            _orchestrator = MultiAgentOrchestrator(
                progress_callback=_on_progress
            )
    return _orchestrator


def get_qa_engine() -> QAEngine:
    global _qa_engine
    if _qa_engine is None:
        retriever = VectorRetriever()
        retriever.load()   # load existing index if available
        _qa_engine = QAEngine(retriever=retriever)
    return _qa_engine


# ══════════════════════════════════════════════════════════════════════════
# Routes — Pages
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/agents")
def agents_page():
    """
    Report Generator page.
    Shows every analysed repository and the reports available to download.
    Reuses the existing download endpoints — no duplicate logic.
    """
    from flask import make_response

    orch = get_orchestrator()
    # Also load any repos analysed in previous sessions
    if not orch.shared_knowledge.all_repos():
        orch.shared_knowledge.load_from_disk()

    # Build a list of repos with their available report files
    repos = []
    for full_name in orch.shared_knowledge.all_repos():
        owner, name = _split_full_name(full_name)
        repo_dir = os.path.join(OUTPUT_DIR, f"{owner}__{name}")
        reports = {
            "knowledge":           os.path.exists(os.path.join(repo_dir, "knowledge.json")),
            "report":              os.path.exists(os.path.join(repo_dir, "report.md")),
            "architecture_report": os.path.exists(os.path.join(repo_dir, "architecture.json")),
        }
        # Only include if at least one report exists
        if any(reports.values()):
            repos.append({
                "full_name": full_name,
                "name":      name,
                "owner":     owner,
                "reports":   reports,
            })

    # Also scan OUTPUT_DIR for repos not yet in shared_knowledge
    if os.path.isdir(OUTPUT_DIR):
        for entry in os.scandir(OUTPUT_DIR):
            if not entry.is_dir():
                continue
            full_name = entry.name.replace("__", "/", 1)
            if any(r["full_name"] == full_name for r in repos):
                continue
            owner, name = _split_full_name(full_name)
            reports = {
                "knowledge":           os.path.exists(os.path.join(entry.path, "knowledge.json")),
                "report":              os.path.exists(os.path.join(entry.path, "report.md")),
                "architecture_report": os.path.exists(os.path.join(entry.path, "architecture.json")),
            }
            if any(reports.values()):
                repos.append({
                    "full_name": full_name,
                    "name":      name,
                    "owner":     owner,
                    "reports":   reports,
                })

    resp = make_response(render_template("agents.html", repos=repos))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/chat")
def chat():
    # Ensure index is loaded (from disk if needed) before rendering the page
    orch = get_orchestrator()
    qa   = get_qa_engine()
    qa.ensure_index(shared_knowledge=orch.shared_knowledge)

    # Show "no knowledge" banner only when the engine truly has no chunks
    has_knowledge = qa.has_knowledge()
    repos = list(orch.shared_knowledge.all_repos())
    return render_template("chat.html", repos=repos, has_knowledge=has_knowledge)


@app.route("/discovery")
def discovery():
    from flask import make_response
    ranking    = _pipeline_result.get("agent1_result", {}).get("ranking", {})
    validation = _pipeline_result.get("agent1_result", {}).get("validation", {})
    resp = make_response(
        render_template("discovery.html", ranking=ranking, validation=validation)
    )
    # Prevent the browser from caching this page so every visit reflects
    # the latest pipeline_result written by /api/discover.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/dashboard")
def dashboard():
    from flask import make_response
    full_name = request.args.get("repo", "")
    orch = get_orchestrator()

    # Ensure any disk-saved knowledge is loaded into shared_knowledge
    if not orch.shared_knowledge.all_repos():
        orch.shared_knowledge.load_from_disk()

    # 1. Try shared_knowledge (populated by api_analyze background thread)
    data = orch.shared_knowledge.get(full_name) if full_name else None

    # 2. Fallback: try _pipeline_result["per_repo"] (same thread-safe dict)
    if data is None and full_name:
        data = _pipeline_result.get("per_repo", {}).get(full_name)

    # Build repos list from every available source
    repos = list(orch.shared_knowledge.all_repos())
    for fn in _pipeline_result.get("per_repo", {}):
        if fn not in repos:
            repos.append(fn)

    resp = make_response(
        render_template("dashboard.html", repo=full_name, data=data, repos=repos)
    )
    # Prevent bfcache from serving a stale pre-analysis render when the user
    # navigates back from Architecture → Dashboard.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/architecture")
def architecture():
    from flask import make_response
    full_name = request.args.get("repo", "")
    owner, name = _split_full_name(full_name)
    agent2_results = SemanticCodeGraphAgent.load_results(owner, name) if owner else {}
    arch = agent2_results.get("architecture", {})
    resp = make_response(
        render_template("architecture.html", repo=full_name, arch=arch)
    )
    # Prevent bfcache from serving a stale pre-analysis render when the user
    # navigates back from Dashboard → Architecture.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


# ══════════════════════════════════════════════════════════════════════════
# Routes — API
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/validate-urls", methods=["POST"])
def api_validate_urls():
    data = request.get_json(force=True)
    urls = data.get("urls", [])
    result = validate_url_list(urls)
    return jsonify(result)


@app.route("/api/discover", methods=["POST", "OPTIONS"])
def api_discover():
    """
    Fast discovery: validate → GitHub metadata → keyword filter → rank.
    No cloning, no code analysis. Returns ranking immediately (synchronous).
    This is what the 'Find & Analyze' button on the home page calls.
    """
    global _pipeline_result

    # Handle CORS pre-flight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data     = request.get_json(force=True)
    urls     = data.get("urls", [])
    keywords = data.get("keywords", "")

    try:
        orch   = get_orchestrator()
        result = orch.run_discovery(urls=urls, keywords=keywords)

        # Store in _pipeline_result so /discovery page can read it
        with _lock:
            _pipeline_result = {
                "success":       result["success"],
                "agent1_result": {
                    "ranking":    result.get("ranking", {}),
                    "validation": result.get("validation", {}),
                    "analyses":   [],
                },
                "per_repo":    {},
                "elapsed_sec": result.get("elapsed_sec", 0),
            }

        if not result["success"]:
            return jsonify({"error": result.get("error", "Discovery failed.")}), 400

        return jsonify({
            "success": True,
            "ranking": result.get("ranking", {}),
        })

    except Exception as exc:
        logger.exception("Discovery error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Deep per-repository analysis.  Each repository tracks its own running
    state independently — clicking Analyze for Spark while Airflow is still
    being analysed starts Spark immediately without any 409.

    Returns {"started": True} within ~1 s.
    Poll /api/status?repo=<full_name> for per-repo progress.
    """
    import time as _time

    data      = request.get_json(force=True)
    full_name = data.get("full_name", "")
    urls      = data.get("urls", [])
    keywords  = data.get("keywords", "")

    with _lock:
        st = _get_repo_status(full_name)

        if st["running"]:
            # ── Stale-state guard ──────────────────────────────────────
            started_at = st.get("started_at")
            if started_at and (_time.monotonic() - started_at) > _MAX_ANALYSIS_SECONDS:
                logger.warning("Stale lock for %s — auto-clearing.", full_name)
                st["running"] = False
                st["error"]   = "Previous analysis timed out and was auto-cleared."
            else:
                # Same repo is genuinely still running
                return jsonify({
                    "error":        "Analysis already running for this repository.",
                    "current_repo": full_name,
                }), 409

        # Initialise a fresh status entry for this repo
        _repo_status[full_name] = {
            "running":          True,
            "steps":            [],
            "error":            None,
            "analysis_started": True,
            "clone_done":       False,
            "agent1_done":      False,
            "agent2_done":      False,
            "rag_done":         False,
            "started_at":       _time.monotonic(),
            "current_repo":     full_name,
        }

    def _prog(step: str, msg: str = ""):
        _on_progress(step, msg, repo=full_name)

    def run():
        global _qa_engine
        import time
        try:
            orch = get_orchestrator()

            if full_name:
                owner, name = _split_full_name(full_name)

                _prog("Fetching repository metadata", full_name)
                from services.github_service import fetch_repo_metadata
                meta = fetch_repo_metadata(f"https://github.com/{full_name}")

                _prog("Cloning repository", full_name)
                from services.repository_cloner import clone_repository
                clone = clone_repository(owner, name, meta.get("clone_url", ""))
                print(f"[CodeLens] {full_name} clone done")

                if not clone["success"]:
                    with _lock:
                        _repo_status[full_name]["error"] = clone["message"]
                    return

                with _lock:
                    _repo_status[full_name]["clone_done"] = True
                _prog("Clone complete — starting source code analysis", full_name)

                _prog("Agent 1 — scanning and analysing source code", full_name)
                a1_result = orch._agent1.analyse_single(
                    owner=owner, name=name,
                    local_path=clone["local_path"],
                    metadata=meta,
                )
                print(f"[CodeLens] {full_name} agent1 done")

                a1_summary = {
                    "metadata":    meta,
                    "statistics":  a1_result.get("statistics", {}),
                    "knowledge":   a1_result.get("knowledge"),
                    "report_path": a1_result.get("report_path", ""),
                }

                orch.shared_knowledge.store(full_name, a1_summary, {})
                with _lock:
                    if "per_repo" not in _pipeline_result:
                        _pipeline_result["per_repo"] = {}
                    _pipeline_result["per_repo"][full_name] = {
                        "agent1": a1_summary,
                        "agent2": {},
                    }
                    _repo_status[full_name]["agent1_done"] = True

                _prog("Agent 1 complete — code knowledge ready", full_name)

                _prog("Agent 2 — building semantic code graph", full_name)
                a2_result = orch._agent2.run(
                    owner=owner, name=name,
                    knowledge=a1_result.get("knowledge"),
                )
                print(f"[CodeLens] {full_name} agent2 done")

                orch.shared_knowledge.store(full_name, a1_summary, a2_result)
                with _lock:
                    _pipeline_result["per_repo"][full_name]["agent2"] = a2_result
                    _repo_status[full_name]["agent2_done"] = True

                _prog("Agent 2 complete — architecture graph ready", full_name)

                _prog("Building RAG index", full_name)
                _qa_engine = None
                qa = get_qa_engine()
                qa.build_index(orch.shared_knowledge)
                print(f"[CodeLens] {full_name} rag done")
                with _lock:
                    _repo_status[full_name]["rag_done"] = True

                _prog("Analysis complete", full_name)

            else:
                # Legacy full-pipeline mode
                result = orch.run_full_pipeline(urls=urls, keywords=keywords)
                with _lock:
                    _pipeline_result.update(result)
                _qa_engine = None
                qa = get_qa_engine()
                if result.get("success"):
                    qa.build_index(orch.shared_knowledge)

        except Exception as exc:
            logger.exception("Analyze error for %s: %s", full_name, exc)
            with _lock:
                if full_name in _repo_status:
                    _repo_status[full_name]["error"] = str(exc)
        finally:
            with _lock:
                if full_name in _repo_status:
                    _repo_status[full_name]["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/analysis/reset", methods=["POST"])
def api_analysis_reset():
    """
    Force-clear the running state for a specific repository.
    Body: {"full_name": "owner/repo"}  (optional; clears all if omitted)
    """
    data      = request.get_json(force=True) or {}
    full_name = data.get("full_name", "")

    with _lock:
        if full_name:
            st = _repo_status.get(full_name)
            if not st or not st["running"]:
                return jsonify({"reset": False, "message": "Not running."})
            st["running"] = False
            st["error"]   = st.get("error") or "Analysis was reset."
        else:
            # Legacy: reset the most recently started running repo
            cleared = False
            for fn, st in _repo_status.items():
                if st.get("running"):
                    st["running"] = False
                    st["error"]   = st.get("error") or "Analysis was reset."
                    cleared = True
            if not cleared:
                return jsonify({"reset": False, "message": "Not running."})

    return jsonify({"reset": True})


@app.route("/api/status")
def api_status():
    """
    Return analysis status for a specific repository.
    Query param: ?repo=owner/repo

    If no repo is given, returns a summary of all currently running analyses.
    """
    full_name = request.args.get("repo", "")

    with _lock:
        if full_name:
            st = _repo_status.get(full_name, _fresh_repo_status(full_name))
            running = st["running"]
            error   = st["error"]
            done    = (not running and st.get("analysis_started", False) and not error)
            return jsonify({
                "running":      running,
                "steps":        st["steps"][-20:],
                "error":        error,
                "done":         done,
                "current_repo": full_name,
                "clone_done":   st.get("clone_done",  False),
                "agent1_done":  st.get("agent1_done", False),
                "agent2_done":  st.get("agent2_done", False),
                "rag_done":     st.get("rag_done",    False),
            })
        else:
            # No specific repo — return list of all running repos (legacy compat)
            running_repos = [
                fn for fn, st in _repo_status.items() if st.get("running")
            ]
            return jsonify({
                "running":      bool(running_repos),
                "running_repos": running_repos,
                "steps":        [],
                "error":        None,
                "done":         False,
                "current_repo": running_repos[0] if running_repos else "",
                "clone_done":   False,
                "agent1_done":  False,
                "agent2_done":  False,
                "rag_done":     False,
            })


@app.route("/api/results")
def api_results():
    if not _pipeline_result:
        return jsonify({"error": "No results yet."}), 404
    # Return a lightweight summary (not the full knowledge JSON)
    a1 = _pipeline_result.get("agent1_result", {})
    return jsonify({
        "success":    _pipeline_result.get("success"),
        "ranking":    a1.get("ranking", {}),
        "validation": a1.get("validation", {}),
        "elapsed":    _pipeline_result.get("elapsed_sec"),
    })


@app.route("/api/ranking")
def api_ranking():
    """
    Return the latest discovery ranking directly.
    The Discovery page calls this on load as a fallback so it can
    render even if the server-side template rendered before
    _pipeline_result was populated (browser cache issue).
    """
    a1      = _pipeline_result.get("agent1_result", {})
    ranking = a1.get("ranking", {})
    if not ranking:
        return jsonify({"error": "No ranking available."}), 404
    return jsonify(ranking)


@app.route("/api/repo/<path:full_name>/stats")
def api_repo_stats(full_name):
    orch = get_orchestrator()
    data = orch.shared_knowledge.get(full_name)
    if not data:
        # Try direct pipeline result
        data = _pipeline_result.get("per_repo", {}).get(full_name)
    if not data:
        return jsonify({"error": "Repository not found."}), 404
    stats = data.get("agent1", {}).get("statistics", {})
    return jsonify(stats)


@app.route("/api/repo/<path:full_name>/knowledge")
def api_repo_knowledge(full_name):
    """Return the full Agent 1 knowledge JSON."""
    from services.knowledge_generator import load_knowledge
    owner, name = _split_full_name(full_name)
    knowledge = load_knowledge(owner, name)
    if not knowledge:
        return jsonify({"error": "Knowledge not found. Run Agent 1 first."}), 404
    return jsonify(knowledge)


@app.route("/api/repo/<path:full_name>/architecture")
def api_repo_architecture(full_name):
    owner, name = _split_full_name(full_name)
    results = SemanticCodeGraphAgent.load_results(owner, name)
    if not results.get("architecture"):
        return jsonify({"error": "Architecture analysis not found. Run Agent 2 first."}), 404
    return jsonify(results["architecture"])


@app.route("/api/run-agent2/<path:full_name>", methods=["POST"])
def api_run_agent2(full_name):
    """Run Agent 2 for a single repo (on-demand)."""
    owner, name = _split_full_name(full_name)

    def run():
        orch = get_orchestrator()
        knowledge = None
        # Try shared knowledge first
        sk = orch.shared_knowledge.get(full_name)
        if sk:
            knowledge = sk.get("agent1", {}).get("knowledge")
        result = orch.run_agent2_for_repo(owner=owner, name=name, knowledge=knowledge)
        if result.get("success") and sk:
            sk["agent2"] = result
        elif result.get("success"):
            orch.shared_knowledge.store(full_name, {}, result)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400

    qa = get_qa_engine()
    # Auto-load index from disk / shared_knowledge if not already loaded
    qa.ensure_index(shared_knowledge=get_orchestrator().shared_knowledge)

    answer = qa.query(question)
    return jsonify(answer)


# ── Download endpoints ────────────────────────────────────────

@app.route("/api/download/<path:full_name>/knowledge")
def download_knowledge(full_name):
    owner, name = _split_full_name(full_name)
    path = os.path.join(OUTPUT_DIR, f"{owner}__{name}", "knowledge.json")
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404
    return send_file(path, as_attachment=True, download_name=f"{name}_knowledge.json")


@app.route("/api/download/<path:full_name>/report")
def download_report(full_name):
    owner, name = _split_full_name(full_name)
    path = os.path.join(OUTPUT_DIR, f"{owner}__{name}", "report.md")
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404
    return send_file(path, as_attachment=True, download_name=f"{name}_report.md")


@app.route("/api/download/<path:full_name>/graph")
def download_graph(full_name):
    """Raw graph JSON — kept for Agent 2 / RAG internal use."""
    owner, name = _split_full_name(full_name)
    path = os.path.join(GRAPH_DIR, f"{owner}__{name}", "graph.json")
    if not os.path.exists(path):
        return jsonify({"error": "Graph not found."}), 404
    return send_file(path, as_attachment=True, download_name=f"{name}_graph.json")


@app.route("/api/download/<path:full_name>/architecture-report")
def download_architecture_report(full_name):
    """
    Human-readable architecture report derived from Agent 2 analysis.
    Shows components, layers, relationships, and entry points in plain text.
    """
    import tempfile
    owner, name = _split_full_name(full_name)

    # Load architecture JSON saved by Agent 2
    arch_path = os.path.join(OUTPUT_DIR, f"{owner}__{name}", "architecture.json")
    if not os.path.exists(arch_path):
        return jsonify({"error": "Architecture analysis not found. Run Agent 2 first."}), 404

    try:
        from services.graph_builder import build_graph_from_knowledge
        from services.architecture_analyzer import ArchitectureAnalyzer
        from services.knowledge_generator import load_knowledge

        knowledge = load_knowledge(owner, name)
        if not knowledge:
            return jsonify({"error": "Repository knowledge not found."}), 404

        graph    = build_graph_from_knowledge(knowledge)
        analyser = ArchitectureAnalyzer(graph)
        report   = analyser.build_human_readable_report(full_name)

        # Write to a temp file and serve it
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(report)
        tmp.close()
        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=f"{name}_architecture_report.txt",
            mimetype="text/plain",
        )
    except Exception as exc:
        logger.exception("Architecture report generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/graph-html/<path:full_name>")
def serve_graph_html(full_name):
    owner, name = _split_full_name(full_name)
    path = os.path.join(GRAPH_DIR, f"{owner}__{name}", "graph.html")
    if not os.path.exists(path):
        return "<p>Graph not yet generated. Run Agent 2 first.</p>", 404
    return send_file(path)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _split_full_name(full_name: str):
    """Split 'owner/repo' → (owner, repo). Handles encoded slashes."""
    parts = full_name.strip("/").split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return full_name, full_name


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    # On Windows, the Werkzeug reloader uses a subprocess that can cause
    # mid-request restarts and "Failed to fetch" errors in the browser.
    # Disable the reloader; restart manually after code changes.
    use_reloader = not sys.platform.startswith("win")
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True,
        use_reloader=use_reloader,
    )
