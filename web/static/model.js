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
    appendLog(`ML: ${data.ml_acc_before}% → ${data.ml_acc_after}% | Spread: ${data.spread_acc_before}% → ${data.spread_acc_after}%`, "good");

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

  const bs = data.book_stats || {};
  const hasBooks = bs.games_with_odds > 0;

  let html = `<h3 style="margin-bottom:12px;font-size:15px">Learn Results — ${data.date}</h3>`;

  // Summary stats
  html += `<div class="learn-summary">
    <div class="learn-summary-stat">
      <div class="learn-summary-val">${data.games}</div>
      <div class="learn-summary-label">Games</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val">${data.ml_acc_before}%</div>
      <div class="learn-summary-label">ML Before</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val" style="color:var(--green)">${data.ml_acc_after}%</div>
      <div class="learn-summary-label">ML After</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val">${data.spread_acc_before}%</div>
      <div class="learn-summary-label">Spread Before</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val" style="color:var(--green)">${data.spread_acc_after}%</div>
      <div class="learn-summary-label">Spread After</div>
    </div>`;

  if (hasBooks) {
    const spreadPct = Math.round(bs.spread_vs_book / bs.games_with_odds * 100);
    const mlPct = Math.round(bs.ml_vs_book / bs.games_with_odds * 100);
    const ouPct = Math.round(bs.ou_vs_book / bs.games_with_odds * 100);
    html += `
    <div class="learn-summary-stat" style="border-left:2px solid var(--yellow);padding-left:14px">
      <div class="learn-summary-val">${spreadPct}%</div>
      <div class="learn-summary-label">vs Book Spread</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val">${mlPct}%</div>
      <div class="learn-summary-label">vs Book ML</div>
    </div>
    <div class="learn-summary-stat">
      <div class="learn-summary-val">${ouPct}%</div>
      <div class="learn-summary-label">vs Book O/U</div>
    </div>`;
  }
  html += `</div>`;

  if (data.results) {
    html += `<table class="learn-table">
      <thead><tr>
        <th style="text-align:left">Game</th>
        <th>Model</th>
        <th>Book Line</th>
        <th>Actual</th>
        <th>ML</th>
        <th>Spread</th>
        <th>O/U</th>
        <th>vs Book</th>
      </tr></thead><tbody>`;

    for (const r of data.results) {
      const mlCls = r.ml_correct ? "learn-correct" : "learn-wrong";
      const mlIcon = r.ml_correct ? "✓" : "✗";
      const spreadCls = r.spread_correct ? "learn-correct" : "learn-wrong";
      const spreadIcon = r.spread_correct ? "✓" : "✗";
      const ouCls = r.total_error <= 10 ? "learn-correct" : "learn-wrong";

      // Book line display
      let bookLine = "—";
      if (r.book_spread != null || r.book_total != null) {
        const sp = r.book_spread != null ? (r.book_spread > 0 ? "+" : "") + r.book_spread : "—";
        const tt = r.book_total != null ? r.book_total : "—";
        bookLine = `${sp} / ${tt}`;
      }

      // vs Book badges
      let vsBook = "";
      if (r.beat_book_spread != null) {
        vsBook += `<span class="${r.beat_book_spread ? "learn-correct" : "learn-wrong"}" title="Spread vs book">S${r.beat_book_spread ? "✓" : "✗"}</span> `;
      }
      if (r.beat_book_ml != null) {
        vsBook += `<span class="${r.beat_book_ml ? "learn-correct" : "learn-wrong"}" title="ML vs book">M${r.beat_book_ml ? "✓" : "✗"}</span> `;
      }
      if (r.beat_book_ou != null) {
        vsBook += `<span class="${r.beat_book_ou ? "learn-correct" : "learn-wrong"}" title="O/U vs book">O${r.beat_book_ou ? "✓" : "✗"}</span>`;
      }
      if (!vsBook) vsBook = '<span class="learn-na">no odds</span>';

      html += `<tr>
        <td>${r.game}</td>
        <td>${r.pred_margin > 0 ? "+" : ""}${r.pred_margin} / ${r.pred_total}</td>
        <td>${bookLine}</td>
        <td>${r.actual_margin > 0 ? "+" : ""}${r.actual_margin} / ${r.actual_total}</td>
        <td class="${mlCls}">${mlIcon}</td>
        <td class="${spreadCls}">${spreadIcon}</td>
        <td class="${ouCls}">${r.ou_result}</td>
        <td>${vsBook}</td>
      </tr>`;
    }

    html += `</tbody></table>`;
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
  grid.style.cssText = "display:flex;flex-direction:column;gap:12px";

  for (const p of preds) {
    grid.appendChild(buildPredCard(p));
  }

  predContainer.appendChild(grid);
}

/* ── helpers ──────────────────────────────────────────────────────── */

function fmtOdds(n) {
  if (n == null) return null;
  return n > 0 ? `+${n}` : `${n}`;
}

function fmtSpread(n) {
  if (n == null) return null;
  const v = parseFloat(n);
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function betStrength(conf) {
  if (conf >= 65) return "strong";
  if (conf >= 57) return "moderate";
  return "weak";
}

function confBarHtml(pct) {
  // Normalize: 50% = 0 fill, 80% = 100% fill (so bar shows "edge above coinflip")
  const fill = Math.min(100, Math.max(0, (pct - 50) / 30 * 100));
  return `<div class="pred-conf-track"><div class="pred-conf-fill" style="width:${fill.toFixed(0)}%"></div></div>`;
}

function buildPredCard(p) {
  const card = document.createElement("div");
  card.className = "pred-card";

  const homeWP  = parseFloat(p.win_prob);          // home win %
  const awayWP  = parseFloat((100 - homeWP).toFixed(1));
  const margin  = p.predicted_margin;               // home − away (positive = home wins)
  const total   = p.predicted_total;
  // ML direction is driven by win_prob (the classification head), not by the
  // margin regression head — the two NN heads can disagree slightly.
  const homeFav = homeWP >= 50;

  /* ── Header ─────────────────────────────────────────────────────── */
  const awayLogo = p.away_logo ? `<img src="${p.away_logo}" class="pred-team-logo" onerror="this.style.display='none'">` : "";
  const homeLogo = p.home_logo ? `<img src="${p.home_logo}" class="pred-team-logo" onerror="this.style.display='none'">` : "";

  let html = `
    <div class="pred-header">
      <div class="pred-header-top">
        <div class="pred-teams">
          ${awayLogo}${p.away_name || p.away_team}
          <span class="pred-at">@</span>
          ${homeLogo}${p.home_name || p.home_team}
        </div>
      </div>
      <div class="pred-wp-row">
        <span class="pred-wp-label left ${!homeFav ? "active" : ""}">${p.away_team} ${awayWP.toFixed(1)}%</span>
        <div class="pred-wp-track">
          <div class="pred-wp-away ${!homeFav ? "fav" : ""}" style="width:${awayWP.toFixed(1)}%"></div>
          <div class="pred-wp-home ${homeFav ? "fav" : ""}" style="width:${homeWP.toFixed(1)}%"></div>
        </div>
        <span class="pred-wp-label right ${homeFav ? "active" : ""}">${homeWP.toFixed(1)}% ${p.home_team}</span>
      </div>
    </div>
    <div class="pred-bets">`;

  /* ── ML row ──────────────────────────────────────────────────────── */
  const mlTeam    = homeFav ? p.home_team : p.away_team;
  const mlOddsNum = homeFav ? p.odds_home_ml : p.odds_away_ml;
  const mlConf    = Math.max(homeWP, 100 - homeWP);
  const mlStr     = betStrength(mlConf);
  const mlOddsTxt = mlOddsNum != null ? `<b>${fmtOdds(mlOddsNum)}</b>` : "";
  const oppOddsNum = homeFav ? p.odds_away_ml : p.odds_home_ml;
  const oppTxt    = oppOddsNum != null ? ` · Opp ${fmtOdds(oppOddsNum)}` : "";

  html += `
    <div class="pred-bet-row ${mlStr}">
      <div class="pred-bet-label">ML</div>
      <div class="pred-bet-body">
        <div class="pred-bet-pick">${mlTeam} to Win</div>
        <div class="pred-bet-sub">Market: ${mlOddsTxt || "—"}${oppTxt} · Model: ${homeWP.toFixed(1)}% home / ${awayWP.toFixed(1)}% away</div>
      </div>
      <div class="pred-bet-conf">
        <span class="pred-conf-pct">${mlConf.toFixed(1)}%</span>
        ${confBarHtml(mlConf)}
      </div>
    </div>`;

  /* ── Spread row ──────────────────────────────────────────────────── */
  if (p.odds_spread != null) {
    // odds_spread = home team's spread (e.g. -5.5 means home is the 5.5-pt favourite).
    // Correct cover formula: edge = margin + homeSpread
    //   homeSpread = -5.5 → home must win by >5.5 to cover → edge = margin - 5.5
    //   Positive edge → home covers; negative edge → away covers.
    const homeSpread = parseFloat(p.odds_spread);
    const awaySpread = -homeSpread;
    const edge       = margin + homeSpread;          // positive → home covers
    const absEdge    = Math.abs(edge);
    const homeCovers = edge > 0;
    // Covering the spread requires winning by a margin, which is strictly harder
    // than just winning → cap spread confidence at ML confidence when same team.
    let spreadConf = Math.min(80, 52 + absEdge * 2.5);
    if (homeCovers === homeFav) spreadConf = Math.min(spreadConf, mlConf);
    const spreadStr  = betStrength(spreadConf);
    const pickTeam   = homeCovers ? p.home_team : p.away_team;
    const pickLine   = homeCovers ? fmtSpread(homeSpread) : fmtSpread(awaySpread);
    const edgeTxt    = `${edge > 0 ? "+" : ""}${edge.toFixed(1)} pt edge`;

    html += `
      <div class="pred-bet-row ${spreadStr}">
        <div class="pred-bet-label">Spread</div>
        <div class="pred-bet-body">
          <div class="pred-bet-pick">${pickTeam} ${pickLine}</div>
          <div class="pred-bet-sub">
            Market: <b>${p.home_team} ${fmtSpread(homeSpread)} / ${p.away_team} ${fmtSpread(awaySpread)}</b>
            · Model: ${margin > 0 ? "+" : ""}${margin} · ${edgeTxt}
          </div>
        </div>
        <div class="pred-bet-conf">
          <span class="pred-conf-pct">${spreadConf.toFixed(1)}%</span>
          ${confBarHtml(spreadConf)}
        </div>
      </div>`;
  }

  /* ── Total row ───────────────────────────────────────────────────── */
  if (p.odds_total != null) {
    const mktTotal  = parseFloat(p.odds_total);
    const te        = total - mktTotal;
    const absTe     = Math.abs(te);
    const totalConf = Math.min(75, 52 + absTe * 1.5);
    const totalStr  = betStrength(totalConf);
    const goOver    = te > 0;
    const edgeTxt   = `${te > 0 ? "+" : ""}${te.toFixed(1)} pts vs line`;

    html += `
      <div class="pred-bet-row ${totalStr}">
        <div class="pred-bet-label">Total</div>
        <div class="pred-bet-body">
          <div class="pred-bet-pick">${goOver ? "OVER" : "UNDER"} ${mktTotal}</div>
          <div class="pred-bet-sub">
            Market O/U: <b>${mktTotal}</b>
            · Model total: ${total} · ${edgeTxt}
          </div>
        </div>
        <div class="pred-bet-conf">
          <span class="pred-conf-pct">${totalConf.toFixed(1)}%</span>
          ${confBarHtml(totalConf)}
        </div>
      </div>`;
  }

  /* ── No odds notice ──────────────────────────────────────────────── */
  if (p.odds_spread == null && p.odds_total == null && p.odds_home_ml == null) {
    html += `<div class="pred-no-odds">No sportsbook lines available for this game.</div>`;
  }

  html += `</div>`; // .pred-bets

  /* ── Model raw output summary ────────────────────────────────────── */
  const predWinner  = margin >= 0 ? (p.home_name || p.home_team) : (p.away_name || p.away_team);
  const predMarginAbs = Math.abs(margin).toFixed(1);
  html += `
    <div class="pred-model-summary">
      <span>Model: <b>${predWinner}</b> wins by <b>${predMarginAbs}</b> pts</span>
      <span>Total: <b>${p.predicted_total}</b></span>
    </div>`;

  /* ── Injury note ─────────────────────────────────────────────────── */
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
