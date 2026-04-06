/* ── Model Page ───────────────────────────────────────────────────── */

const statusDot      = document.getElementById("status-dot");
const statusTitle    = document.getElementById("status-title");
const statusSubtitle = document.getElementById("status-subtitle");
const btnTrain       = document.getElementById("btn-train");
const btnPredict     = document.getElementById("btn-predict");
const btnUpdate      = document.getElementById("btn-update-data");
const trainLog       = document.getElementById("train-log");
const predContainer  = document.getElementById("predictions-container");
const predCount      = document.getElementById("pred-count");

let pollTimer = null;

/* ── Status ───────────────────────────────────────────────────────── */

async function loadStatus() {
  try {
    const r = await fetch("/api/model/status");
    const s = await r.json();
    renderStatus(s);
  } catch {
    statusTitle.textContent = "Error loading status";
    statusDot.className = "status-dot inactive";
  }
}

function renderStatus(s) {
  // Dot + title
  if (s.training_in_progress) {
    statusDot.className = "status-dot training";
    statusTitle.textContent = "Training in Progress";
    statusSubtitle.textContent = "Model is currently training...";
    btnTrain.disabled = true;
    btnPredict.disabled = true;
    showLog();
    showProgressBar();
    startPollTraining();
    return;
  }

  if (s.trained) {
    const dt = s.trained_at ? new Date(s.trained_at).toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
    }) : "";
    statusDot.className = "status-dot active";
    statusTitle.textContent = "Model Active";
    statusSubtitle.textContent = `Trained ${dt}`;
    btnTrain.disabled = false;
    btnPredict.disabled = false;
  } else {
    statusDot.className = "status-dot inactive";
    statusTitle.textContent = "No Model Trained";
    statusSubtitle.textContent = "Train the model to generate predictions";
    btnTrain.disabled = false;
    btnPredict.disabled = true;
  }

  // Data freshness
  if (s.data_status === "behind") {
    btnUpdate.style.display = "";
    statusSubtitle.textContent += ` · Data last updated: ${s.last_data_date || "unknown"}`;
  } else {
    btnUpdate.style.display = "none";
  }

  // Metrics
  if (s.metrics) {
    const m = s.metrics;
    setText("metric-accuracy",  m.win_accuracy  != null ? (m.win_accuracy * 100).toFixed(1) + "%" : "—");
    setText("metric-margin",    m.margin_mae    != null ? m.margin_mae.toFixed(1) + " pts" : "—");
    setText("metric-total",     m.total_mae     != null ? m.total_mae.toFixed(1) + " pts" : "—");
    setText("metric-direction", m.margin_direction != null ? (m.margin_direction * 100).toFixed(1) + "%" : "—");
    setText("metric-logloss",   m.log_loss      != null ? m.log_loss.toFixed(4) : "—");
    setText("metric-samples",   m.train_samples != null ? m.train_samples.toLocaleString() : "—");
  }
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

/* ── Training ─────────────────────────────────────────────────────── */

btnTrain.addEventListener("click", async () => {
  if (!confirm("Start model training? This may take several minutes.")) return;
  btnTrain.disabled = true;
  btnPredict.disabled = true;
  statusDot.className = "status-dot training";
  statusTitle.textContent = "Training in Progress";
  statusSubtitle.textContent = "Model is training...";
  showLog();
  showProgressBar();
  appendLog("Starting model training...", "info");

  try {
    await fetch("/api/model/train", { method: "POST" });
    startPollTraining();
  } catch (e) {
    appendLog("Failed to start training: " + e.message, "error");
    btnTrain.disabled = false;
  }
});

function showLog() {
  trainLog.classList.add("visible");
}

function appendLog(msg, cls = "") {
  const div = document.createElement("div");
  div.className = "log-line" + (cls ? " " + cls : "");
  div.textContent = msg;
  trainLog.appendChild(div);
  trainLog.scrollTop = trainLog.scrollHeight;
}

function startPollTraining() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTrainingStatus, 2000);
}

async function pollTrainingStatus() {
  try {
    const r = await fetch("/api/model/train/status");
    const s = await r.json();

    // Show new log lines
    const existingCount = trainLog.querySelectorAll(".log-line").length;
    const lines = s.log || [];
    for (let i = existingCount; i < lines.length; i++) {
      const line = lines[i];
      let cls = "";
      if (line.includes("Epoch")) cls = "info";
      else if (line.includes("Test Results") || line.includes("COMPLETE")) cls = "good";
      else if (line.includes("ERROR") || line.includes("error")) cls = "error";
      appendLog(line, cls);
    }

    // Update progress bar
    if (s.progress && s.progress.total_games) {
      updateProgressBar(s.progress);
    }

    if (s.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      appendLog("Training complete!", "good");
      hideProgressBar();
      loadStatus();
    } else if (s.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      appendLog("Training failed: " + (s.error || "unknown error"), "error");
      hideProgressBar();
      btnTrain.disabled = false;
      statusDot.className = "status-dot inactive";
      statusTitle.textContent = "Training Failed";
    }
  } catch {
    // Network error, keep polling
  }
}

/* ── Progress Bar ────────────────────────────────────────────────── */

const progressContainer = document.getElementById("progress-container");
const progressBar       = document.getElementById("progress-bar");
const progressLabel     = document.getElementById("progress-label");
const progressEta       = document.getElementById("progress-eta");
const progressGames     = document.getElementById("progress-games");
const progressStats     = document.getElementById("progress-stats");

function showProgressBar() {
  progressContainer.style.display = "";
}

function hideProgressBar() {
  progressContainer.style.display = "none";
}

function updateProgressBar(p) {
  showProgressBar();
  progressBar.style.width = p.pct + "%";
  progressLabel.textContent = p.phase === "TEST" ? "Testing..." : "Training...";
  progressEta.textContent = p.eta_display ? `ETA: ${p.eta_display}` : "";
  progressGames.textContent = `${p.game_idx.toLocaleString()} / ${p.total_games.toLocaleString()} games`;

  const parts = [];
  if (p.loss != null) parts.push(`Loss: ${p.loss}`);
  if (p.train_acc != null) parts.push(`Acc: ${p.train_acc}%`);
  if (p.games_per_sec != null) parts.push(`${p.games_per_sec} games/s`);
  progressStats.textContent = parts.join(" · ");
}

/* ── Data Update ──────────────────────────────────────────────────── */

btnUpdate.addEventListener("click", async () => {
  btnUpdate.disabled = true;
  btnUpdate.innerHTML = '<span class="model-spinner"></span> Updating...';
  showLog();
  appendLog("Updating data...", "info");

  try {
    const r = await fetch("/api/model/update-data", { method: "POST" });
    const s = await r.json();
    if (s.ok) {
      appendLog("Data updated successfully!", "good");
      btnUpdate.style.display = "none";
      loadStatus();
    } else {
      appendLog("Data update failed: " + (s.error || "unknown"), "error");
    }
  } catch (e) {
    appendLog("Data update request failed: " + e.message, "error");
  } finally {
    btnUpdate.disabled = false;
    btnUpdate.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
        <path d="M21 3v5h-5"/>
        <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
        <path d="M8 16H3v5"/>
      </svg>
      Update Data`;
  }
});

/* ── Learn from Results ────────────────────────────────────────────── */

const btnLearn = document.getElementById("btn-learn");
const learnResults = document.getElementById("learn-results");

btnLearn.addEventListener("click", async () => {
  btnLearn.disabled = true;
  btnLearn.innerHTML = '<span class="model-spinner"></span> Learning...';
  showLog();
  appendLog("Fetching yesterday's results and fine-tuning...", "info");

  try {
    const r = await fetch("/api/model/learn", { method: "POST" });
    const data = await r.json();

    if (data.error) {
      appendLog("Learn error: " + data.error, "error");
      return;
    }

    appendLog(`Learned from ${data.games} games`, "good");
    appendLog(`Accuracy: ${data.accuracy_before}% → ${data.accuracy_after}%`, "good");

    renderLearnResults(data);
    loadStatus(); // refresh metrics
  } catch (e) {
    appendLog("Learn request failed: " + e.message, "error");
  } finally {
    btnLearn.disabled = false;
    btnLearn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
      </svg>
      Learn from Yesterday`;
  }
});

function renderLearnResults(data) {
  learnResults.classList.add("visible");

  let html = `
    <h3 style="margin-bottom:12px;font-size:15px">Learn Results — ${data.date}</h3>
    <div class="learn-summary">
      <div class="learn-summary-stat">
        <div class="learn-summary-val">${data.games}</div>
        <div class="learn-summary-label">Games</div>
      </div>
      <div class="learn-summary-stat">
        <div class="learn-summary-val">${data.accuracy_before}%</div>
        <div class="learn-summary-label">Before</div>
      </div>
      <div class="learn-summary-stat">
        <div class="learn-summary-val" style="color:var(--green)">${data.accuracy_after}%</div>
        <div class="learn-summary-label">After</div>
      </div>
    </div>`;

  if (data.results) {
    for (const r of data.results) {
      const cls = r.correct ? "learn-correct" : "learn-wrong";
      const icon = r.correct ? "✓" : "✗";
      html += `
        <div class="learn-game-row">
          <span>${r.game}</span>
          <span>Pred: ${r.pred_margin > 0 ? "+" : ""}${r.pred_margin} / ${r.pred_total}</span>
          <span>Actual: ${r.actual_margin > 0 ? "+" : ""}${r.actual_margin} / ${r.actual_total}</span>
          <span class="${cls}">${icon} Err: ${r.margin_error}pts</span>
        </div>`;
    }
  }

  learnResults.innerHTML = html;
}


/* ── Predictions ──────────────────────────────────────────────────── */

btnPredict.addEventListener("click", async () => {
  btnPredict.disabled = true;
  btnPredict.innerHTML = '<span class="model-spinner"></span> Loading...';

  try {
    const r = await fetch("/api/model/predict");
    const preds = await r.json();
    renderPredictions(preds);
  } catch (e) {
    predContainer.innerHTML = `<div class="model-empty"><p>Failed to load predictions: ${e.message}</p></div>`;
  } finally {
    btnPredict.disabled = false;
    btnPredict.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
      </svg>
      Get Today's Picks`;
  }
});

function renderPredictions(preds) {
  if (!preds || preds.length === 0) {
    predContainer.innerHTML = '<div class="model-empty"><p>No games found for today.</p></div>';
    predCount.style.display = "none";
    return;
  }

  predCount.textContent = preds.length;
  predCount.style.display = "";

  predContainer.innerHTML = "";
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px";

  for (const p of preds) {
    grid.appendChild(buildPredCard(p));
  }

  predContainer.appendChild(grid);
}

function buildPredCard(p) {
  const card = document.createElement("div");
  card.className = "pred-card";

  const confLevel = p.confidence >= 60 ? "high" : p.confidence >= 40 ? "medium" : "low";

  // Matchup header
  let html = `
    <div class="pred-matchup">
      <div class="pred-teams">${p.away_team} @ ${p.home_team}</div>
      <div class="pred-confidence ${confLevel}">${p.confidence}%</div>
    </div>
    <div class="pred-stats">
      <div class="pred-stat">
        <div class="pred-stat-val">${p.win_prob}%</div>
        <div class="pred-stat-label">Home Win %</div>
      </div>
      <div class="pred-stat">
        <div class="pred-stat-val">${p.predicted_margin > 0 ? "+" : ""}${p.predicted_margin}</div>
        <div class="pred-stat-label">Pred Margin</div>
      </div>
      <div class="pred-stat">
        <div class="pred-stat-val">${p.predicted_total}</div>
        <div class="pred-stat-label">Pred Total</div>
      </div>
    </div>`;

  // Picks
  if (p.picks && p.picks.length > 0) {
    html += '<div class="pred-picks">';
    for (const pick of p.picks) {
      const strength = pick.confidence >= 60 ? "strong" : pick.confidence >= 50 ? "moderate" : "";
      html += `
        <div class="pred-pick ${strength}">
          <div>
            <span class="pred-pick-name">${pick.pick}</span>
            <span class="pred-pick-type">${pick.type}</span>
          </div>
          <span class="pred-pick-conf">${pick.confidence}%</span>
        </div>`;
    }
    html += '</div>';
  }

  // Odds chips
  const chips = [];
  if (p.odds_spread != null)  chips.push(`Spread: ${p.odds_spread > 0 ? "+" : ""}${p.odds_spread}`);
  if (p.odds_total != null)   chips.push(`O/U: ${p.odds_total}`);
  if (p.odds_home_ml != null) chips.push(`ML: ${p.odds_home_ml > 0 ? "+" : ""}${p.odds_home_ml}`);
  if (chips.length) {
    html += '<div class="pred-odds">';
    for (const c of chips) html += `<span class="pred-odds-chip">${c}</span>`;
    html += '</div>';
  }

  // Injury note
  if (p.home_missing > 0 || p.away_missing > 0) {
    const parts = [];
    if (p.home_missing > 0) parts.push(`${p.home_team}: ${p.home_missing} out`);
    if (p.away_missing > 0) parts.push(`${p.away_team}: ${p.away_missing} out`);
    html += `<div class="pred-injury-note">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      ${parts.join(" · ")}
    </div>`;
  }

  card.innerHTML = html;
  return card;
}

/* ── Init ─────────────────────────────────────────────────────────── */
loadStatus();
