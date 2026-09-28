/* CodeLens AI — Main JavaScript */

const MAX_REPOS = 10;
const MIN_REPOS = 5;

/* ══════════════════════════════════════════════════════
   URL Management
   ══════════════════════════════════════════════════════ */

function addUrlRow(value = "") {
  const list  = document.getElementById("url-list");
  if (!list) return;
  const count = list.querySelectorAll(".url-row").length;
  if (count >= MAX_REPOS) {
    showUrlError(`Maximum ${MAX_REPOS} repositories allowed.`);
    return;
  }

  const row   = document.createElement("div");
  row.className = "url-row";
  row.innerHTML = `
    <span class="url-num" style="color:var(--text2);font-size:.82rem;min-width:22px">${count + 1}</span>
    <input
      type="url"
      class="url-input"
      placeholder="https://github.com/owner/repository"
      value="${escHtml(value)}"
      oninput="onUrlInput(this)"
    />
    <span class="dup-badge hidden" title="Duplicate URL — will be ignored">DUP</span>
    <button class="remove-btn" onclick="removeUrlRow(this)" title="Remove">✕</button>
  `;
  list.appendChild(row);
  updateUrlCount();
  clearUrlErrors();
}

function removeUrlRow(btn) {
  const list = document.getElementById("url-list");
  btn.closest(".url-row").remove();
  // Renumber
  list.querySelectorAll(".url-row").forEach((row, i) => {
    const num = row.querySelector(".url-num");
    if (num) num.textContent = i + 1;
  });
  // Re-run duplicate detection after removal
  markDuplicates();
  updateUrlCount();
}

/** Normalize a URL the same way the backend does, for client-side dup detection. */
function normalizeUrl(url) {
  return url.trim().replace(/\/+$/, "").replace(/\.git$/, "").toLowerCase();
}

/**
 * Walk all URL inputs, mark duplicates with the DUP badge,
 * highlight their input field in warning colour, and return
 * the set of normalized unique URLs.
 */
function markDuplicates() {
  const inputs  = Array.from(document.querySelectorAll(".url-input"));
  const seen    = new Map();   // normalizedUrl → first input element

  // First pass: build seen map from top to bottom
  inputs.forEach(input => {
    const val  = input.value.trim();
    const norm = val ? normalizeUrl(val) : null;
    const row  = input.closest(".url-row");
    const badge = row ? row.querySelector(".dup-badge") : null;

    // Reset state
    input.classList.remove("duplicate");
    if (badge) badge.classList.add("hidden");

    if (!norm || !val) return;

    if (seen.has(norm)) {
      // This row is the duplicate — mark it
      input.classList.add("duplicate");
      if (badge) badge.classList.remove("hidden");
    } else {
      seen.set(norm, input);
    }
  });

  return seen;   // keyed by normalizedUrl, value = first (kept) input
}

function updateUrlCount() {
  const seen   = markDuplicates();   // re-run detection on every update
  const unique = seen.size;          // only count non-duplicate, non-empty URLs

  const msg   = document.getElementById("url-count-msg");
  if (msg) {
    msg.textContent = `${unique} / ${MAX_REPOS} repositories`;
    msg.style.color = "var(--text2)";
  }

  // Show/hide the duplicate warning banner
  const list   = document.getElementById("url-list");
  const inputs = list ? list.querySelectorAll(".url-input") : [];
  const totalFilled = Array.from(inputs).filter(i => i.value.trim()).length;
  const dupCount    = totalFilled - unique;

  const dupWarn = document.getElementById("dup-warning");
  if (dupWarn) {
    if (dupCount > 0) {
      dupWarn.textContent =
        `${dupCount} duplicate URL${dupCount > 1 ? "s" : ""} detected — ` +
        `${unique} unique ${unique === 1 ? "repository" : "repositories"} will be analyzed.`;
      dupWarn.classList.remove("hidden");
    } else {
      dupWarn.classList.add("hidden");
    }
  }

  const addBtn = document.getElementById("add-url-btn");
  const rowCount = list ? list.querySelectorAll(".url-row").length : 0;
  if (addBtn) addBtn.disabled = rowCount >= MAX_REPOS;
}

function onUrlInput(input) {
  const val = input.value.trim();
  // Format validation (red border for bad URLs)
  const formatOk = !val || /^https?:\/\/(www\.)?github\.com\/[\w.\-]+\/[\w.\-]+(\.git)?\/?$/.test(val);
  input.classList.toggle("invalid", !formatOk);
  // Re-run duplicate detection whenever any input changes
  updateUrlCount();
}

/** Return only the unique, non-empty, valid-format URLs (in input order, first occurrence wins). */
function getUrls() {
  const inputs = Array.from(document.querySelectorAll(".url-input"));
  const seen   = new Set();
  const result = [];
  for (const input of inputs) {
    const val  = input.value.trim();
    if (!val) continue;
    const norm = normalizeUrl(val);
    if (seen.has(norm)) continue;   // silently skip duplicates
    seen.add(norm);
    result.push(val);
  }
  return result;
}

/* ══════════════════════════════════════════════════════
   Analysis Pipeline — calls /api/discover (fast, synchronous)
   ══════════════════════════════════════════════════════ */

async function startAnalysis() {
  clearUrlErrors();
  const urls     = getUrls();          // already deduplicated
  const keywords = (document.getElementById("keywords-input")?.value || "").trim();

  // Client-side validation (format errors only — duplicates are already filtered)
  const errors = [];
  if (urls.length === 0) errors.push("Please add at least one repository URL.");
  if (urls.length > MAX_REPOS) errors.push(`Maximum ${MAX_REPOS} URLs allowed.`);

  for (const u of urls) {
    if (!/^https?:\/\/(www\.)?github\.com\//i.test(u)) {
      errors.push(`Not a GitHub URL: ${escHtml(u)}`);
    }
  }

  if (errors.length) { showUrlErrors(errors); return; }

  // Show progress panel
  showProgress(true);
  addStep(`Validating ${urls.length} unique ${urls.length === 1 ? "repository" : "repositories"} …`, false);

  const analyzeBtn = document.getElementById("analyze-btn");
  if (analyzeBtn) analyzeBtn.disabled = true;

  // Quick server health check before the main request
  try {
    const ping = await fetch("/api/health", {method: "GET"});
    if (!ping.ok) throw new Error("health check failed");
  } catch (_) {
    showUrlErrors([
      "Cannot reach the CodeLens AI server. "
      + "Make sure Flask is running: open a terminal in the project folder and run "
      + "<code>python app.py</code>, then open <strong>http://127.0.0.1:5000/</strong>"
    ]);
    showProgress(false);
    if (document.getElementById("analyze-btn"))
      document.getElementById("analyze-btn").disabled = false;
    return;
  }

  try {
    addStep("Collecting repository metadata …", false);
    const resp = await fetch("/api/discover", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ urls, keywords }),
    });

    // Check we actually got a response (not a network-level failure)
    if (!resp) {
      throw new Error("No response from server.");
    }

    let data;
    try {
      data = await resp.json();
    } catch (_) {
      throw new Error(`Server returned an unexpected response (HTTP ${resp.status}).`);
    }

    if (!resp.ok || data.error) {
      showUrlErrors([data.error || `Server error (HTTP ${resp.status}). Please try again.`]);
      showProgress(false);
      if (analyzeBtn) analyzeBtn.disabled = false;
      return;
    }

    addStep("Filtering and ranking repositories …", true);
    addStep("Discovery complete! Redirecting …", true);

    const spinner = document.getElementById("progress-spinner");
    if (spinner) spinner.style.display = "none";

    // Use replace() so back button doesn't re-trigger the form.
    // 1200ms gives the server time to finish writing _pipeline_result
    // before the browser's GET /discovery request arrives.
    setTimeout(() => { window.location.replace("/discovery"); }, 1200);

  } catch (err) {
    // Translate the opaque "Failed to fetch" into something actionable
    let msg = String(err);
    if (msg.includes("Failed to fetch") || msg.includes("NetworkError") ||
        msg.includes("fetch") || msg.includes("network")) {
      msg = "Cannot reach the CodeLens AI server. "
          + "Make sure Flask is running (python app.py) on port 5000 "
          + "and open the app at http://127.0.0.1:5000/";
    }
    showUrlErrors([msg]);
    showProgress(false);
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

// pollStatus kept for background operations
let _pollTimer = null;

function pollStatus() {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(async () => {
    try {
      const resp = await fetch("/api/status");
      const data = await resp.json();

      const stepsEl = document.getElementById("progress-steps");
      if (stepsEl && data.steps) {
        stepsEl.innerHTML = "";
        data.steps.forEach(s => {
          const div = document.createElement("div");
          div.className = "step-item done";
          div.innerHTML = `<span class="step-icon">✓</span> ${escHtml(s.step)}${s.msg ? ' — ' + escHtml(s.msg) : ''}`;
          stepsEl.appendChild(div);
        });
      }

      if (data.error) {
        clearInterval(_pollTimer);
        showUrlErrors([`Error: ${data.error}`]);
        showProgress(false);
        const btn = document.getElementById("analyze-btn");
        if (btn) btn.disabled = false;
        return;
      }

      if (data.done) {
        clearInterval(_pollTimer);
        const spinner = document.getElementById("progress-spinner");
        if (spinner) spinner.style.display = "none";
        addStep("Complete! Redirecting …", true);
        setTimeout(() => { window.location.href = "/dashboard"; }, 1000);
      }
    } catch (_) { /* ignore network hiccups */ }
  }, 1200);
}

/* ══════════════════════════════════════════════════════
   UI helpers
   ══════════════════════════════════════════════════════ */

function showProgress(show) {
  const panel = document.getElementById("progress-panel");
  if (panel) panel.classList.toggle("hidden", !show);
}

function addStep(msg, done = false) {
  const el = document.getElementById("progress-steps");
  if (!el) return;
  const div = document.createElement("div");
  div.className = "step-item" + (done ? " done" : "");
  div.innerHTML = `<span class="step-icon">${done ? "✓" : "…"}</span> ${escHtml(msg)}`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function showUrlError(msg) { showUrlErrors([msg]); }

function showUrlErrors(msgs) {
  const el = document.getElementById("url-errors");
  if (!el) return;
  el.classList.remove("hidden");
  el.innerHTML = "<ul>" + msgs.map(m => `<li>${m}</li>`).join("") + "</ul>";
}

function clearUrlErrors() {
  const el = document.getElementById("url-errors");
  if (el) { el.classList.add("hidden"); el.innerHTML = ""; }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
