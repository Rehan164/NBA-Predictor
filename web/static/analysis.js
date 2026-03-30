/* NBA Predictor — Analysis Page */

let pollInterval = null;
let logOffset = 0;

const runBtn    = document.getElementById("run-btn");
const logEl     = document.getElementById("progress-log");
const picksSec  = document.getElementById("picks-section");
const picksList = document.getElementById("picks-list");
const picksCount = document.getElementById("picks-count");
const parlayBody = document.getElementById("parlay-body");
const fullAnalysis = document.getElementById("full-analysis");
const toggleBtn  = document.getElementById("toggle-analysis-btn");

// ── Helpers ───────────────────────────────────────────────────────────────
function probClass(p) {
  if (p >= 60) return "high";
  if (p >= 52) return "medium";
  return "low";
}

function formatOdds(odds) {
  if (!odds) return "—";
  return odds > 0 ? `+${odds}` : `${odds}`;
}

// ── Log rendering ─────────────────────────────────────────────────────────
function appendLog(lines) {
  lines.slice(logOffset).forEach(line => {
    const div = document.createElement("div");
    div.className = "log-line" +
      (line.startsWith("ERROR") ? " error" : "") +
      (line === "Done." ? " done" : "");
    div.textContent = `> ${line}`;
    logEl.appendChild(div);
  });
  logOffset = lines.length;
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Render picks ──────────────────────────────────────────────────────────
function renderPicks(picks) {
  picksCount.textContent = picks.length;
  picksList.innerHTML = picks.map((pick, i) => {
    const cls = probClass(pick.probability);
    return `
      <div class="pick-card">
        <div class="pick-number">${i + 1}</div>
        <div class="pick-desc">${pick.description}</div>
        <div class="pick-prob-block">
          <span class="pick-prob ${cls}">${pick.probability}%</span>
          <span class="pick-odds">${formatOdds(pick.odds)}</span>
        </div>
      </div>`;
  }).join("");
}

// ── Render parlay table ───────────────────────────────────────────────────
function renderParlay(legs) {
  if (!legs.length) return;

  const pickDescriptions = Array.from(picksList.querySelectorAll(".pick-desc"))
    .map(el => el.textContent.trim());

  parlayBody.innerHTML = legs.map((leg, i) => {
    const cls = probClass(leg.probability);
    const desc = pickDescriptions.slice(0, leg.legs).join(", ");
    const barWidth = Math.max(4, leg.probability);

    return `
      <tr>
        <td style="font-weight:600;color:var(--text)">${leg.legs}-leg</td>
        <td style="color:var(--text-muted);font-size:12px;max-width:300px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${desc}</td>
        <td>
          <div class="parlay-prob-bar">
            <div class="prob-bar-track">
              <div class="prob-bar-fill ${cls}" style="width:${barWidth}%"></div>
            </div>
            <span style="font-weight:600;color:var(--text);min-width:40px">${leg.probability}%</span>
          </div>
        </td>
        <td style="font-weight:700;color:${cls === 'high' ? 'var(--green)' : cls === 'medium' ? 'var(--yellow)' : 'var(--red)'}">
          ${formatOdds(leg.odds)}
        </td>
      </tr>`;
  }).join("");
}

// ── Poll for status ───────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res = await fetch("/api/analysis/status");
    const data = await res.json();

    appendLog(data.log || []);

    if (data.status === "done") {
      stopPolling();
      runBtn.disabled = false;
      runBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Run Again`;

      if (data.result) {
        const { picks, parlay_legs, analysis } = data.result;
        renderPicks(picks);
        renderParlay(parlay_legs);
        fullAnalysis.textContent = analysis || "";
        picksSec.classList.add("visible");
      }
    } else if (data.status === "error") {
      stopPolling();
      runBtn.disabled = false;
      runBtn.textContent = "Retry";
    }
  } catch (e) {
    console.error("Poll error:", e);
  }
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

// ── Run analysis ──────────────────────────────────────────────────────────
async function startAnalysis() {
  // Reset
  await fetch("/api/analysis/reset", { method: "POST" });
  logEl.innerHTML = "";
  logOffset = 0;
  picksSec.classList.remove("visible");
  fullAnalysis.classList.remove("visible");
  toggleBtn.textContent = "Show full Claude analysis";
  logEl.classList.add("visible");

  const targetPicks = parseInt(document.getElementById("picks-target").value) || 5;

  runBtn.disabled = true;
  runBtn.innerHTML = `
    <svg class="refresh-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/>
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
      <path d="M8 16H3v5"/>
    </svg>
    Running...`;

  try {
    await fetch("/api/analysis/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ picks: targetPicks }),
    });
    pollInterval = setInterval(pollStatus, 1500);
  } catch (e) {
    runBtn.disabled = false;
    runBtn.textContent = "Run Analysis";
  }
}

// ── Events ────────────────────────────────────────────────────────────────
runBtn.addEventListener("click", startAnalysis);

toggleBtn.addEventListener("click", () => {
  const visible = fullAnalysis.classList.toggle("visible");
  toggleBtn.innerHTML = visible
    ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg> Hide full analysis`
    : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg> Show full Claude analysis`;
});

// ── On load: check if analysis already ran ────────────────────────────────
(async () => {
  try {
    const res = await fetch("/api/analysis/status");
    const data = await res.json();
    if (data.status === "done" && data.result) {
      logOffset = 0;
      logEl.classList.add("visible");
      appendLog(data.log || []);
      const { picks, parlay_legs, analysis } = data.result;
      renderPicks(picks);
      renderParlay(parlay_legs);
      fullAnalysis.textContent = analysis || "";
      picksSec.classList.add("visible");
      runBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Run Again`;
    } else if (data.status === "running") {
      logEl.classList.add("visible");
      runBtn.disabled = true;
      runBtn.textContent = "Running...";
      pollInterval = setInterval(pollStatus, 1500);
    }
  } catch {}
})();
