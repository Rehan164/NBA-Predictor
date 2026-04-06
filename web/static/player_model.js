/* ── Player Model Page ─────────────────────────────────────────────── */

const pmDot      = document.getElementById("pm-dot");
const pmTitle    = document.getElementById("pm-title");
const pmSubtitle = document.getElementById("pm-subtitle");
const btnTrain   = document.getElementById("btn-train");
const btnPredict = document.getElementById("btn-predict");
const pmLog      = document.getElementById("pm-log");
const pmPreds    = document.getElementById("pm-predictions");

let pollTimer = null;

/* ── Status ─────────────────────────────────────────────────────────── */

async function loadStatus() {
  try {
    const r = await fetch("/api/player-model/status");
    const s = await r.json();
    renderStatus(s);
  } catch {
    pmTitle.textContent = "Error loading status";
    pmDot.className = "pm-status-dot inactive";
  }
}

function renderStatus(s) {
  if (s.training_in_progress) {
    pmDot.className = "pm-status-dot training";
    pmTitle.textContent = "Training in Progress";
    pmSubtitle.textContent = "Player props model is training...";
    btnTrain.disabled = true;
    btnPredict.disabled = true;
    showLog();
    startPoll();
    return;
  }

  if (s.trained) {
    const dt = s.trained_at ? new Date(s.trained_at).toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
    }) : "";
    pmDot.className = "pm-status-dot active";
    pmTitle.textContent = "Player Model Active";
    pmSubtitle.textContent = `Trained ${dt} | ${s.ensemble_size || 5} models | ${(s.train_samples || 0).toLocaleString()} samples`;
    btnTrain.disabled = false;
    btnPredict.disabled = false;
  } else {
    pmDot.className = "pm-status-dot inactive";
    pmTitle.textContent = "No Player Model Trained";
    pmSubtitle.textContent = "Train the model to predict player props";
    btnTrain.disabled = false;
    btnPredict.disabled = true;
  }

  if (s.metrics) {
    const m = s.metrics;
    setText("m-pts-mae", m.pts ? m.pts.mae + " pts" : "--");
    setText("m-reb-mae", m.reb ? m.reb.mae + " pts" : "--");
    setText("m-ast-mae", m.ast ? m.ast.mae + " pts" : "--");
    setText("m-pts-w3",  m.pts ? m.pts.within_3 + "%" : "--");
    setText("m-reb-w3",  m.reb ? m.reb.within_3 + "%" : "--");
    setText("m-ast-w3",  m.ast ? m.ast.within_3 + "%" : "--");
  }
  setText("m-ensemble", s.ensemble_size || "--");
  setText("m-features", s.feature_count || "--");
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

/* ── Training ────────────────────────────────────────────────────────── */

btnTrain.addEventListener("click", async () => {
  if (!confirm("Start player props model training? This may take several minutes.")) return;
  btnTrain.disabled = true;
  btnPredict.disabled = true;
  pmDot.className = "pm-status-dot training";
  pmTitle.textContent = "Training in Progress";
  pmSubtitle.textContent = "Training player props ensemble...";
  showLog();
  appendLog("Starting player props model training...", "info");

  try {
    await fetch("/api/player-model/train", { method: "POST" });
    startPoll();
  } catch (e) {
    appendLog("Failed to start training: " + e.message, "error");
    btnTrain.disabled = false;
  }
});

function showLog() { pmLog.classList.add("visible"); }

function appendLog(msg, cls = "") {
  const div = document.createElement("div");
  div.className = "log-line" + (cls ? " " + cls : "");
  div.textContent = msg;
  pmLog.appendChild(div);
  pmLog.scrollTop = pmLog.scrollHeight;
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 2000);
}

async function pollStatus() {
  try {
    const r = await fetch("/api/player-model/train/status");
    const s = await r.json();

    const existingCount = pmLog.querySelectorAll(".log-line").length;
    const lines = s.log || [];
    for (let i = existingCount; i < lines.length; i++) {
      const line = lines[i];
      let cls = "";
      if (line.includes("Epoch") || line.includes("Model")) cls = "info";
      else if (line.includes("COMPLETE") || line.includes("MAE")) cls = "good";
      else if (line.includes("ERROR") || line.includes("error")) cls = "error";
      appendLog(line, cls);
    }

    if (s.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      appendLog("Training complete!", "good");
      loadStatus();
    } else if (s.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      appendLog("Training failed: " + (s.error || "unknown"), "error");
      btnTrain.disabled = false;
      pmDot.className = "pm-status-dot inactive";
      pmTitle.textContent = "Training Failed";
    }
  } catch { /* keep polling */ }
}

/* ── Predictions ─────────────────────────────────────────────────────── */

btnPredict.addEventListener("click", async () => {
  btnPredict.disabled = true;
  btnPredict.innerHTML = '<span class="pm-spinner"></span> Loading...';

  try {
    const r = await fetch("/api/player-model/predict");
    const games = await r.json();
    renderPredictions(games);
  } catch (e) {
    pmPreds.innerHTML = `<div class="pm-empty"><p>Failed: ${e.message}</p></div>`;
  } finally {
    btnPredict.disabled = false;
    btnPredict.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
      </svg>
      Predict Today's Props`;
  }
});

function renderPredictions(games) {
  if (!games || games.length === 0 || games[0]?.error) {
    pmPreds.innerHTML = `<div class="pm-empty"><p>${games[0]?.error || "No games found for today."}</p></div>`;
    return;
  }

  pmPreds.innerHTML = "";

  for (const game of games) {
    const card = document.createElement("div");
    card.className = "pm-game";

    // Format game time
    const gt = new Date(game.game_time);
    const timeStr = gt.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });

    // Header
    card.innerHTML = `
      <div class="pm-game-header">
        <div class="pm-game-teams">
          <img src="${game.away_logo}" alt="" onerror="this.style.display='none'">
          <span>${game.away_team}</span>
          <span style="color:var(--text-muted);font-weight:400">@</span>
          <img src="${game.home_logo}" alt="" onerror="this.style.display='none'">
          <span>${game.home_team}</span>
        </div>
        <div class="pm-game-time">${timeStr}</div>
      </div>`;

    // Player table
    const table = document.createElement("table");
    table.className = "pm-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>Player</th>
          <th>Min</th>
          <th>Pred PTS</th>
          <th>Avg PTS</th>
          <th>Pred REB</th>
          <th>Avg REB</th>
          <th>Pred AST</th>
          <th>Avg AST</th>
        </tr>
      </thead>`;

    const tbody = document.createElement("tbody");

    // Group by team
    const awayPlayers = game.players.filter(p => p.team === game.away_team);
    const homePlayers = game.players.filter(p => p.team === game.home_team);

    for (const [teamName, players] of [[game.away_team, awayPlayers], [game.home_team, homePlayers]]) {
      // Team separator
      const sepRow = document.createElement("tr");
      sepRow.className = "pm-team-row";
      sepRow.innerHTML = `<td colspan="8">${teamName}</td>`;
      tbody.appendChild(sepRow);

      // Sort by predicted points
      players.sort((a, b) => (b.pred_pts || 0) - (a.pred_pts || 0));

      for (const p of players) {
        const row = document.createElement("tr");
        if (p.is_out) {
          row.style.opacity = "0.4";
        }

        const injBadge = p.is_out ? '<span class="pm-injury-badge out">OUT</span>' : '';

        row.innerHTML = `
          <td>
            <div class="pm-player-cell">
              <div>
                <div class="pm-player-name">${p.name}${injBadge}</div>
                <div class="pm-player-team">${p.team}</div>
              </div>
            </div>
          </td>
          <td>${p.is_out ? '--' : p.avg_min || '--'}</td>
          <td><span class="pm-pred-val">${p.is_out ? '--' : p.pred_pts}</span></td>
          <td><span class="pm-avg-val">${p.avg_pts || '--'}</span> ${diffBadge(p.pred_pts, p.avg_pts, p.is_out)}</td>
          <td><span class="pm-pred-val">${p.is_out ? '--' : p.pred_reb}</span></td>
          <td><span class="pm-avg-val">${p.avg_reb || '--'}</span> ${diffBadge(p.pred_reb, p.avg_reb, p.is_out)}</td>
          <td><span class="pm-pred-val">${p.is_out ? '--' : p.pred_ast}</span></td>
          <td><span class="pm-avg-val">${p.avg_ast || '--'}</span> ${diffBadge(p.pred_ast, p.avg_ast, p.is_out)}</td>`;

        tbody.appendChild(row);
      }
    }

    table.appendChild(tbody);
    card.appendChild(table);
    pmPreds.appendChild(card);
  }
}

function diffBadge(pred, avg, isOut) {
  if (isOut || pred == null || avg == null || avg === '--') return '';
  const diff = pred - avg;
  if (Math.abs(diff) < 0.3) return '<span class="pm-diff flat">--</span>';
  const cls = diff > 0 ? "up" : "down";
  const sign = diff > 0 ? "+" : "";
  return `<span class="pm-diff ${cls}">${sign}${diff.toFixed(1)}</span>`;
}

/* ── Init ────────────────────────────────────────────────────────────── */
loadStatus();
