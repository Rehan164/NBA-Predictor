/* NBA Predictor — Dashboard JS */

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  activeStat: "pts",
  players: [],
  props: {},
  modalPlayer: null,
  modalStat: "pts",
  modalLine: null,
  modalPropOdds: null, // {line, over, under} from Odds API
  chart: null,
  games: [],        // full game objects (with event_id + predictions)
};

// ── Helpers ───────────────────────────────────────────────────────────────
function formatTime(isoStr) {
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" }) + " ET";
  } catch {
    return "";
  }
}

function confidenceBadge(val) {
  if (!val) return "";
  const c = parseFloat(val);
  if (c >= 60) return `<span class="pred-badge high">${c.toFixed(0)}%</span>`;
  if (c >= 55) return `<span class="pred-badge med">${c.toFixed(0)}%</span>`;
  return `<span class="pred-badge low">${c.toFixed(0)}%</span>`;
}

function hitRateClass(rate) {
  if (rate >= 65) return "good";
  if (rate >= 45) return "ok";
  return "poor";
}

function getStatLabel(stat) {
  return { pts: "PTS", reb: "REB", ast: "AST" }[stat] || stat.toUpperCase();
}

// ── Fetch ─────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ── Games ─────────────────────────────────────────────────────────────────
async function loadGames() {
  const container = document.getElementById("games-container");
  container.innerHTML = `<div class="state-loading"><div class="spinner"></div><span>Loading games...</span></div>`;

  try {
    const games = await fetchJSON("/api/games");
    state.games = games;
    renderGames(games);
    document.getElementById("games-count").textContent = games.length;
  } catch (e) {
    container.innerHTML = `<div class="state-empty"><span class="state-empty-icon">—</span><span>Could not load games</span></div>`;
  }
}

function renderGames(games) {
  const container = document.getElementById("games-container");

  if (!games.length) {
    container.innerHTML = `<div class="state-empty"><span class="state-empty-icon">—</span><span>No games today</span></div>`;
    return;
  }

  container.innerHTML = games.map(g => {
    const preds = g.predictions || {};
    const ml = preds.moneyline || {};
    const sp = preds.spread || {};
    const tot = preds.totals || {};

    const isLive = g.status === "STATUS_IN_PROGRESS";
    const isFinal = g.status === "STATUS_FINAL";

    let timeHtml = "";
    if (isLive) {
      timeHtml = `<div class="game-status-live"><span class="live-dot"></span>${g.status_display || "LIVE"}</div>`;
    } else if (isFinal) {
      timeHtml = `<div class="game-time">FINAL</div>`;
    } else {
      timeHtml = `<div class="game-time">${formatTime(g.game_time)}</div>`;
    }

    const showScore = isLive || isFinal;

    const awayBlock = `
      <div class="team-block">
        ${g.away_logo ? `<img class="team-logo" src="${g.away_logo}" alt="${g.away_team}" onerror="this.style.display='none'">` : ""}
        <span class="team-abbr">${g.away_team}</span>
        ${showScore ? `<span class="team-score">${g.away_score}</span>` : ""}
      </div>`;

    const homeBlock = `
      <div class="team-block">
        ${g.home_logo ? `<img class="team-logo" src="${g.home_logo}" alt="${g.home_team}" onerror="this.style.display='none'">` : ""}
        <span class="team-abbr">${g.home_team}</span>
        ${showScore ? `<span class="team-score">${g.home_score}</span>` : ""}
      </div>`;

    const mkt = g.market || {};

    // Detect model vs market disagreement on ML
    let disagreement = false;
    if (ml.pick && mkt.away_ml && mkt.home_ml) {
      const awayOdds = parseInt(mkt.away_ml);
      const homeOdds = parseInt(mkt.home_ml);
      const mktFavored = awayOdds < homeOdds ? g.away_team : g.home_team;
      if (ml.pick !== mktFavored) disagreement = true;
    }

    // Market odds row
    let mktHtml = "";
    if (mkt.away_ml || mkt.spread) {
      const awayFav = mkt.away_ml && parseInt(mkt.away_ml) < 0;
      const homeFav = mkt.home_ml && parseInt(mkt.home_ml) < 0;
      const awayMl  = mkt.away_ml  ? `<span class="mkt-odds" style="color:${awayFav ? "var(--green)" : "var(--text-dim)"}">${mkt.away_ml}</span>`  : "";
      const homeMl  = mkt.home_ml  ? `<span class="mkt-odds" style="color:${homeFav ? "var(--green)" : "var(--text-dim)"}">${mkt.home_ml}</span>`  : "";
      const spread  = mkt.spread   ? `<span class="mkt-val">${g.away_team} ${mkt.spread}</span>` : "";
      const total   = mkt.total    ? `<span class="mkt-val">${mkt.total}</span>` : "";
      mktHtml = `
        <div class="game-divider"></div>
        <div class="market-odds-row">
          <div class="market-col">
            <span class="market-label">DK Odds</span>
            <div class="market-ml-pair">
              <span class="mkt-team">${g.away_team}</span>${awayMl}
              <span class="mkt-sep">·</span>
              <span class="mkt-team">${g.home_team}</span>${homeMl}
            </div>
          </div>
          <div class="market-col right">
            ${spread ? `<span class="market-label">Spread</span>${spread}` : ""}
            ${total  ? `<span class="market-label">O/U</span>${total}` : ""}
          </div>
        </div>`;
    }

    let predsHtml = "";
    if (Object.keys(preds).length) {
      const mlPick = ml.pick
        ? `${ml.pick} ${confidenceBadge(ml.confidence)}${disagreement ? ' <span class="disagree-flag" title="Model disagrees with market">⚠</span>' : ""}`
        : "—";

      // ── Spread: model vs market ──────────────────────────────────────────
      // sp.predicted_margin = home_score − away_score (positive = home wins)
      // mkt.spread          = away team's line  (e.g. "+8.5" → away gets 8.5 pts)
      // edge = model_margin − spread_away
      //   positive → home covers market line
      //   negative → away covers market line
      const modelMargin = sp.predicted_margin;
      const spLine = sp.favored
        ? `${sp.favored} ${modelMargin >= 0 ? "-" : "+"}${sp.by_points}`
        : "—";

      let spreadEdgeHtml = "";
      let mktSpreadInline = "";
      const mktSpreadRaw = mkt.spread;
      if (modelMargin !== undefined && mktSpreadRaw != null && mktSpreadRaw !== "") {
        const mktSpreadNum = parseFloat(mktSpreadRaw);
        if (!isNaN(mktSpreadNum)) {
          // Show market line next to model (away-team perspective, e.g. "CHA +8.5")
          const mktSpreadFmt = mktSpreadNum >= 0 ? `+${mktSpreadNum.toFixed(1)}` : mktSpreadNum.toFixed(1);
          mktSpreadInline = `<span class="mkt-inline" title="Market line">${g.away_team} ${mktSpreadFmt}</span>`;

          // Edge calculation
          const edge    = modelMargin - mktSpreadNum;
          const absEdge = Math.abs(edge);
          if (absEdge >= 1.5) {
            const homeCovers  = edge > 0;
            const mktHomeNum  = -mktSpreadNum;
            const homeFmt     = mktHomeNum >= 0 ? `+${mktHomeNum.toFixed(1)}` : mktHomeNum.toFixed(1);
            const betLabel    = homeCovers
              ? `${g.home_team} ${homeFmt}`
              : `${g.away_team} ${mktSpreadFmt}`;
            const edgeCls = absEdge >= 4 ? "edge-strong" : "edge-mod";
            spreadEdgeHtml = `<div class="edge-rec ${edgeCls}">▶ ${betLabel}<span class="edge-pts">${absEdge.toFixed(1)} pt edge</span></div>`;
          }
        }
      }

      // ── Total: model vs market ───────────────────────────────────────────
      // edge_total = model_total − market_total
      //   positive → OVER has value   negative → UNDER has value
      const modelTotal = tot.predicted_total;
      const totLine = modelTotal != null ? modelTotal.toFixed(1) : "—";

      let totalEdgeHtml = "";
      let mktTotalInline = "";
      const mktTotalRaw = mkt.total;
      if (modelTotal != null && mktTotalRaw != null && mktTotalRaw !== "") {
        const mktTotalNum = parseFloat(mktTotalRaw);
        if (!isNaN(mktTotalNum)) {
          mktTotalInline = `<span class="mkt-inline" title="Market O/U">${mktTotalNum}</span>`;

          const totalEdge    = modelTotal - mktTotalNum;
          const absTotalEdge = Math.abs(totalEdge);
          if (absTotalEdge >= 1.5) {
            const goOver  = totalEdge > 0;
            const edgeCls = absTotalEdge >= 4 ? "edge-strong" : "edge-mod";
            totalEdgeHtml = `<div class="edge-rec ${edgeCls}">▶ ${goOver ? "OVER" : "UNDER"} ${mktTotalNum}<span class="edge-pts">${absTotalEdge.toFixed(1)} pts</span></div>`;
          }
        }
      }

      predsHtml = `
        <div class="game-divider"></div>
        <div class="prediction-row">
          <span class="pred-label">ML</span>
          <span class="pred-value">${mlPick}</span>
        </div>
        <div class="prediction-row">
          <span class="pred-label">Spread</span>
          <span class="pred-value">${spLine}</span>
          ${mktSpreadInline}
        </div>
        ${spreadEdgeHtml}
        <div class="prediction-row">
          <span class="pred-label">Total</span>
          <span class="pred-value">${totLine}</span>
          ${mktTotalInline}
        </div>
        ${totalEdgeHtml}`;
    }

    // Combine market + model
    const bottomHtml = mktHtml + predsHtml;

    return `
      <div class="game-card${disagreement ? " model-disagrees" : ""}" data-event="${g.event_id}">
        ${timeHtml}
        <div class="game-matchup">
          ${awayBlock}
          <span class="vs-divider">@</span>
          ${homeBlock}
        </div>
        ${bottomHtml}
        <div class="game-card-hint">Tap for rosters</div>
      </div>`;
  }).join("");

  // Attach click handlers
  container.querySelectorAll(".game-card").forEach(card => {
    card.addEventListener("click", () => {
      const eventId = card.dataset.event;
      const game = state.games.find(g => g.event_id === eventId);
      if (game) openGamePanel(game);
    });
  });
}

// ── Players ───────────────────────────────────────────────────────────────
async function loadPlayers() {
  const container = document.getElementById("players-container");
  container.innerHTML = `<div class="state-loading"><div class="spinner"></div><span>Loading players...</span></div>`;

  try {
    state.players = await fetchJSON("/api/players");
    // Load props in background
    loadProps();
    renderPlayers();
    document.getElementById("players-count").textContent = state.players.length;
  } catch (e) {
    container.innerHTML = `<div class="state-empty"><span class="state-empty-icon">—</span><span>Could not load players</span></div>`;
  }
}

async function loadProps() {
  try {
    state.props = await fetchJSON("/api/props");
    if (state.props && !state.props.error) {
      renderPlayers(); // re-render with prop lines
    }
  } catch {
    // Props are optional — fail silently
  }
}

function _findPropEntry(playerName, stat) {
  if (!state.props) return null;

  // Exact match first
  if (state.props[playerName] && state.props[playerName][stat] != null) {
    return state.props[playerName][stat];
  }

  // Partial match (different name formats)
  for (const [propName, data] of Object.entries(state.props)) {
    if (
      propName.toLowerCase().includes(playerName.toLowerCase().split(" ").pop()) ||
      playerName.toLowerCase().includes(propName.toLowerCase().split(" ").pop())
    ) {
      if (data[stat] != null) return data[stat];
    }
  }

  return null;
}

function getPlayerLine(playerName, stat) {
  stat = stat || state.activeStat;
  const entry = _findPropEntry(playerName, stat);
  if (!entry) return null;
  return (entry && typeof entry === "object") ? entry.line : entry;
}

function getPlayerPropData(playerName, stat) {
  stat = stat || state.activeStat;
  return _findPropEntry(playerName, stat);
}

function formatOdds(price) {
  if (price == null) return "";
  return price > 0 ? `+${price}` : `${price}`;
}

function renderPlayers() {
  const container = document.getElementById("players-container");
  const stat = state.activeStat;
  const statLabel = getStatLabel(stat);

  if (!state.players.length) {
    container.innerHTML = `<div class="state-empty"><span class="state-empty-icon">—</span><span>No players found</span></div>`;
    return;
  }

  container.innerHTML = state.players.map(p => {
    const val = p[`avg_${stat}`];
    const line = getPlayerLine(p.name);
    const isHot = p[`hot_${stat}`];

    // Injury badge (shown at bottom, minimal text)
    let injuryHtml = "";
    if (p.is_out) {
      injuryHtml = `<div class="injury-badge out">OUT</div>`;
    } else if (p.is_doubtful) {
      injuryHtml = `<div class="injury-badge doubtful">Doubtful</div>`;
    } else if (p.is_questionable) {
      injuryHtml = `<div class="injury-badge questionable">Questionable</div>`;
    }

    // Line row — always rendered to keep consistent height
    const lineBadgeHtml = `
      <div class="player-line-row">
        <div class="line-badge">${line != null ? `Line: <span class="line-val">${line}</span>` : ""}</div>
      </div>`;

    const headshotHtml = p.headshot
      ? `<img class="player-headshot" src="${p.headshot}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'">`
      : `<div class="player-headshot-placeholder"></div>`;

    const hotBadge = isHot ? `<span class="hot-badge">🔥</span>` : "";
    const statClass = isHot ? "stat-value-big hot" : "stat-value-big";

    return `
      <div class="player-card" data-name="${p.name}" data-team="${p.team}">
        <div class="player-card-top">
          ${headshotHtml}
          <div class="player-card-info">
            <div class="player-header">
              <div class="player-name">${p.name}</div>
              <span class="player-team-badge">${p.team}</span>
            </div>
            <div class="player-matchup">vs ${p.opponent}</div>
            <div class="player-stat-main">
              <span class="${statClass}">${val != null ? val.toFixed(1) : "—"}</span>
              <span class="stat-label-big">${statLabel}</span>
              ${hotBadge}
            </div>
            ${lineBadgeHtml}
          </div>
        </div>
        <div class="player-mini-stats">
          <div class="mini-stat">
            <span class="mini-stat-val">${p.avg_pts != null ? p.avg_pts.toFixed(1) : "—"}${p.hot_pts ? " 🔥" : ""}</span>
            <span class="mini-stat-lbl">PTS</span>
          </div>
          <div class="mini-stat">
            <span class="mini-stat-val">${p.avg_reb != null ? p.avg_reb.toFixed(1) : "—"}${p.hot_reb ? " 🔥" : ""}</span>
            <span class="mini-stat-lbl">REB</span>
          </div>
          <div class="mini-stat">
            <span class="mini-stat-val">${p.avg_ast != null ? p.avg_ast.toFixed(1) : "—"}${p.hot_ast ? " 🔥" : ""}</span>
            <span class="mini-stat-lbl">AST</span>
          </div>
        </div>
        ${injuryHtml}
      </div>`;
  }).join("");

  // Attach click listeners
  container.querySelectorAll(".player-card").forEach(card => {
    card.addEventListener("click", () => {
      const name = card.dataset.name;
      const player = state.players.find(p => p.name === name);
      if (player) openModal(player);
    });
  });
}

// ── Modal ─────────────────────────────────────────────────────────────────
function openModal(player) {
  state.modalPlayer = player;
  state.modalStat = state.activeStat;
  state.modalLine = getPlayerLine(player.name, state.modalStat);
  state.modalPropOdds = getPlayerPropData(player.name, state.modalStat);

  document.getElementById("modal-player-name").textContent = player.name;
  document.getElementById("modal-team").textContent = `${player.team} vs ${player.opponent}`;

  // Injury detail in modal
  const injEl = document.getElementById("modal-injury");
  if (player.is_out || player.is_doubtful || player.is_questionable) {
    const status = player.is_out ? "OUT" : player.is_doubtful ? "Doubtful" : "Questionable";
    const cls    = player.is_out ? "out" : player.is_doubtful ? "doubtful" : "questionable";
    const desc   = player.injury_desc ? ` — ${player.injury_desc}` : "";
    injEl.innerHTML = `<span class="injury-badge ${cls}">${status}</span><span style="font-size:12px;color:var(--text-muted)">${desc}</span>`;
    injEl.style.display = "flex";
  } else {
    injEl.style.display = "none";
    injEl.innerHTML = "";
  }

  const lineInput = document.getElementById("line-input");
  lineInput.value = state.modalLine != null ? state.modalLine : "";
  lineInput.placeholder = "Set line…";

  // Show odds next to line input
  updateOddsDisplay();

  // Set active tab
  document.querySelectorAll(".modal-stat-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.stat === state.modalStat);
  });

  // Show modal
  document.getElementById("modal-backdrop").classList.add("visible");
  document.getElementById("modal-panel").classList.add("visible");
  document.body.style.overflow = "hidden";

  loadPlayerHistory();
}

function updateOddsDisplay() {
  const el = document.getElementById("odds-display");
  if (!el) return;
  const prop = state.modalPropOdds;
  const books = prop?.books;
  if (books && books.length) {
    const first = books[0];
    const overStr = first.over != null ? `O ${formatOdds(first.over)}` : "";
    const underStr = first.under != null ? `U ${formatOdds(first.under)}` : "";
    el.innerHTML = `<span class="odds-book-name">${first.title}</span><span class="odds-over">${overStr}</span><span class="odds-under">${underStr}</span>`;
    el.style.display = "flex";
  } else {
    el.style.display = "none";
    el.innerHTML = "";
  }
}

function closeModal() {
  document.getElementById("modal-backdrop").classList.remove("visible");
  document.getElementById("modal-panel").classList.remove("visible");
  document.body.style.overflow = "";

  if (state.chart) {
    state.chart.destroy();
    state.chart = null;
  }
}

async function loadPlayerHistory() {
  const player = state.modalPlayer;
  const stat = state.modalStat;
  const line = state.modalLine;

  document.getElementById("modal-avg").textContent = "";
  document.getElementById("modal-hit-rate").textContent = "";
  document.getElementById("stats-summary").innerHTML = `
    <div class="summary-card"><div class="summary-val">—</div><div class="summary-lbl">Avg</div></div>
    <div class="summary-card"><div class="summary-val">—</div><div class="summary-lbl">Hit Rate</div></div>
    <div class="summary-card"><div class="summary-val">—</div><div class="summary-lbl">Last 5 Avg</div></div>`;

  try {
    const params = new URLSearchParams({ stat, n: 20 });
    if (line != null) params.set("line", line);

    const data = await fetchJSON(`/api/player/${encodeURIComponent(player.name)}/history?${params}`);
    renderPlayerChart(data);
    renderSummary(data);
  } catch (e) {
    document.getElementById("modal-avg").textContent = "Error loading data";
  }
}

function renderSummary(data) {
  const label = getStatLabel(data.stat);
  const last5 = data.games.slice(-5);
  const last5Avg = last5.length
    ? (last5.reduce((s, g) => s + g.value, 0) / last5.length).toFixed(1)
    : "—";

  let hitHtml = "";
  if (data.hit_rate != null) {
    const cls = hitRateClass(data.hit_rate);
    hitHtml = `<div class="summary-card">
      <div class="summary-val ${cls}">${data.hit_rate}%</div>
      <div class="summary-lbl">${data.hit_count}/${data.total_games} Hit</div>
    </div>`;
  } else {
    hitHtml = `<div class="summary-card">
      <div class="summary-val">—</div>
      <div class="summary-lbl">Hit Rate</div>
    </div>`;
  }

  // Betting line cards from Odds API — one per sportsbook
  let oddsHtml = "";
  const prop = state.modalPropOdds;
  if (prop?.books?.length) {
    oddsHtml = prop.books.map(bk => {
      const overStr = bk.over != null ? `O ${formatOdds(bk.over)}` : "";
      const underStr = bk.under != null ? `U ${formatOdds(bk.under)}` : "";
      const oddsStr = [overStr, underStr].filter(Boolean).join(" / ");
      return `<div class="summary-card summary-card-odds">
        <div class="summary-val">${bk.line}</div>
        <div class="summary-lbl">${bk.title}</div>
        ${oddsStr ? `<div class="summary-odds">${oddsStr}</div>` : ""}
      </div>`;
    }).join("");
  }

  document.getElementById("stats-summary").innerHTML = `
    ${oddsHtml}
    <div class="summary-card">
      <div class="summary-val">${data.avg}</div>
      <div class="summary-lbl">L${data.games.length} Avg ${label}</div>
    </div>
    ${hitHtml}
    <div class="summary-card">
      <div class="summary-val">${last5Avg}</div>
      <div class="summary-lbl">Last 5 Avg</div>
    </div>`;
}

// ── Chart ─────────────────────────────────────────────────────────────────
const BOOK_COLORS = [
  "#6366f1", // indigo (primary)
  "#f59e0b", // amber
  "#06b6d4", // cyan
  "#ec4899", // pink
  "#10b981", // emerald
];

const propLinePlugin = {
  id: "propLine",
  afterDraw(chart) {
    const opts = chart.options.plugins.propLine;
    const books = opts?.books;

    // If we have books from the Odds API, draw one line per book
    if (books?.length) {
      const { ctx, scales: { y, x } } = chart;
      ctx.save();

      books.forEach((bk, i) => {
        if (bk.line == null) return;
        const color = BOOK_COLORS[i % BOOK_COLORS.length];
        const yPx = y.getPixelForValue(bk.line);

        // Dashed line
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(x.left, yPx);
        ctx.lineTo(x.right, yPx);
        ctx.stroke();

        // Label: "DraftKings 25.5 (O -110 / U -110)"
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "right";

        const parts = [];
        if (bk.over != null) parts.push(`O ${bk.over > 0 ? "+" : ""}${bk.over}`);
        if (bk.under != null) parts.push(`U ${bk.under > 0 ? "+" : ""}${bk.under}`);
        const oddsStr = parts.length ? `  (${parts.join(" / ")})` : "";
        ctx.fillText(`${bk.title} ${bk.line}${oddsStr}`, x.right - 4, yPx - 5);
      });

      ctx.restore();
      return;
    }

    // Fallback: single manual line (no books data)
    const line = opts?.value;
    if (line == null) return;

    const { ctx, scales: { y, x } } = chart;
    const yPx = y.getPixelForValue(line);

    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(x.left, yPx);
    ctx.lineTo(x.right, yPx);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#6366f1";
    ctx.font = "bold 11px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${line}`, x.right - 4, yPx - 5);
    ctx.restore();
  },
};

Chart.register(propLinePlugin);

function renderPlayerChart(data) {
  if (state.chart) {
    state.chart.destroy();
    state.chart = null;
  }

  const { games, line, stat } = data;
  const label = getStatLabel(stat);

  const colors = games.map(g => {
    if (g.hit === true)  return "rgba(34,197,94,0.85)";
    if (g.hit === false) return "rgba(239,68,68,0.75)";
    return "rgba(99,102,241,0.6)";
  });

  const borderColors = games.map(g => {
    if (g.hit === true)  return "#22c55e";
    if (g.hit === false) return "#ef4444";
    return "#6366f1";
  });

  const labels = games.map(g => {
    const opp = g.opponent || "";
    return [opp.replace("@", "@ "), g.date];
  });

  const ctx = document.getElementById("player-chart").getContext("2d");

  state.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label,
        data: games.map(g => g.value),
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 5,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(22,22,31,0.95)",
          borderColor: "rgba(99,102,241,0.3)",
          borderWidth: 1,
          titleColor: "#e8e8f0",
          bodyColor: "#9090b0",
          padding: 10,
          callbacks: {
            title(items) {
              const idx = items[0].dataIndex;
              const g = games[idx];
              return `${g.opponent} — ${g.date}`;
            },
            label(item) {
              const g = games[item.dataIndex];
              const hitStr = g.hit === true ? "  Hit" : g.hit === false ? "  Miss" : "";
              return `${label}: ${g.value}${hitStr}`;
            },
          },
        },
        propLine: {
          value: line,
          books: state.modalPropOdds?.books ?? null,
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#6b6b8a",
            font: { size: 10 },
            maxRotation: 0,
          },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: { color: "#6b6b8a", font: { size: 11 } },
          grid: { color: "rgba(255,255,255,0.06)" },
          beginAtZero: false,
        },
      },
    },
  });
}

// ── Game Panel ────────────────────────────────────────────────────────────
function openGamePanel(game) {
  const panel = document.getElementById("game-panel");
  const backdrop = document.getElementById("game-backdrop");
  const inner = document.getElementById("game-panel-inner");

  // Show loading state
  inner.innerHTML = `<div class="state-loading"><div class="spinner"></div><span>Loading rosters...</span></div>`;
  backdrop.classList.add("visible");
  panel.classList.add("visible");
  document.body.style.overflow = "hidden";

  fetchJSON(`/api/game/${game.event_id}/players`)
    .then(data => renderGamePanel(game, data))
    .catch(() => {
      inner.innerHTML = `<div class="state-empty"><span>Could not load rosters</span></div>`;
    });
}

function closeGamePanel() {
  document.getElementById("game-panel").classList.remove("visible");
  document.getElementById("game-backdrop").classList.remove("visible");
  document.body.style.overflow = "";
}

function renderGamePanel(game, data) {
  const inner = document.getElementById("game-panel-inner");
  const preds = game.predictions || {};
  const ml = preds.moneyline || {};
  const sp = preds.spread || {};
  const tot = preds.totals || {};

  const isLive = game.status === "STATUS_IN_PROGRESS";
  const isFinal = game.status === "STATUS_FINAL";
  const timeStr = isLive ? (game.status_display || "LIVE") : isFinal ? "Final" : formatTime(game.game_time);

  const mkt = game.market || {};

  // Prediction pills — market first, then model
  const pills = [];
  if (mkt.away_ml || mkt.home_ml) {
    pills.push(`<div class="gp-pred-pill"><span class="gp-pred-label">DK ML</span><span class="gp-pred-val">${game.away_team} ${mkt.away_ml || "—"} · ${game.home_team} ${mkt.home_ml || "—"}</span></div>`);
  }
  if (mkt.spread)           pills.push(`<div class="gp-pred-pill"><span class="gp-pred-label">Spread</span><span class="gp-pred-val">${game.away_team} ${mkt.spread} (${mkt.spread_odds || ""})</span></div>`);
  if (mkt.total)            pills.push(`<div class="gp-pred-pill"><span class="gp-pred-label">O/U</span><span class="gp-pred-val">${mkt.total}</span></div>`);
  if (ml.pick)              pills.push(`<div class="gp-pred-pill"><span class="gp-pred-label">Model</span><span class="gp-pred-val">${ml.pick} ${ml.confidence ? ml.confidence.toFixed(0)+"%" : ""}</span></div>`);

  const awayLogo = game.away_logo ? `<img class="gp-team-logo" src="${game.away_logo}" onerror="this.style.display='none'">` : "";
  const homeLogo = game.home_logo ? `<img class="gp-team-logo" src="${game.home_logo}" onerror="this.style.display='none'">` : "";

  inner.innerHTML = `
    <div class="game-panel-header">
      <div class="game-panel-title">
        <div class="game-panel-matchup">
          ${awayLogo}
          <span class="gp-vs">${game.away_team}</span>
          <span class="gp-vs">@</span>
          ${homeLogo}
          <span class="gp-vs">${game.home_team}</span>
          <span class="gp-time">${timeStr}</span>
        </div>
        <button class="btn-close" id="game-panel-close">✕</button>
      </div>
      ${pills.length ? `<div class="game-panel-preds">${pills.join("")}</div>` : ""}
    </div>

    <div class="game-teams-grid">
      ${renderTeamRoster(data.away)}
      ${renderTeamRoster(data.home)}
    </div>`;

  document.getElementById("game-panel-close").addEventListener("click", closeGamePanel);
}

function renderTeamRoster(teamData) {
  if (!teamData) return "";
  const { name, logo, players = [] } = teamData;

  const logoHtml = logo ? `<img class="team-roster-logo" src="${logo}" onerror="this.style.display='none'">` : "";

  const starters = players.filter(p => p.starter && !p.is_out);
  const bench    = players.filter(p => !p.starter);

  function playerRow(p) {
    const img = p.headshot
      ? `<img class="roster-headshot" src="${p.headshot}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'">`
      : `<div class="roster-headshot-placeholder"></div>`;

    let injTag = "";
    if (p.is_out)          injTag = `<span class="roster-injury-tag out">OUT</span>`;
    else if (p.is_doubtful) injTag = `<span class="roster-injury-tag doubtful">DTD</span>`;
    else if (p.is_questionable) injTag = `<span class="roster-injury-tag questionable">Q</span>`;

    const starterDot = p.starter ? `<span class="starter-dot" title="Projected starter"></span>` : "";

    // Minutes: avg → predicted
    let minsHtml = "";
    if (p.is_out) {
      minsHtml = `<div class="roster-mins"><span class="min-pred out">OUT</span></div>`;
    } else {
      const diff = p.pred_min - p.avg_min;
      const predClass = diff > 1.5 ? "up" : diff < -1.5 ? "down" : "";
      minsHtml = `
        <div class="roster-mins">
          <span class="min-avg">${p.avg_min}</span>
          <span class="min-arrow">→</span>
          <span class="min-pred ${predClass}">${p.pred_min}</span>
        </div>`;
    }

    return `
      <div class="roster-player ${p.is_out ? "is-out" : ""}">
        ${img}
        <div class="roster-info">
          <div class="roster-name-row">
            ${starterDot}
            <span class="roster-name">${p.name}</span>
            ${injTag}
          </div>
          <span class="roster-stats">${p.avg_pts} PTS · ${p.avg_reb} REB · ${p.avg_ast} AST</span>
        </div>
        ${minsHtml}
      </div>`;
  }

  const startersHtml = starters.length
    ? `<div class="roster-section-label">Projected Starters</div>${starters.map(playerRow).join("")}`
    : "";

  const benchHtml = bench.length
    ? `<div class="roster-section-label">Bench</div>${bench.map(playerRow).join("")}`
    : "";

  return `
    <div class="team-roster-col">
      <div class="team-roster-header">
        ${logoHtml}
        <span class="team-roster-name">${name}</span>
      </div>
      ${startersHtml}
      ${benchHtml}
    </div>`;
}

// ── Event Listeners ───────────────────────────────────────────────────────
function setupEvents() {
  // Stat tabs (players section)
  document.querySelectorAll(".stat-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".stat-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeStat = tab.dataset.stat;
      renderPlayers();
    });
  });

  // Modal stat tabs
  document.querySelectorAll(".modal-stat-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".modal-stat-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      state.modalStat = tab.dataset.stat;
      // Update line from props if available
      if (state.modalPlayer) {
        const propLine = getPlayerLine(state.modalPlayer.name, state.modalStat);
        state.modalLine = propLine;
        state.modalPropOdds = getPlayerPropData(state.modalPlayer.name, state.modalStat);
        document.getElementById("line-input").value = propLine != null ? propLine : "";
        updateOddsDisplay();
      }
      loadPlayerHistory();
    });
  });

  // Line input (apply on Enter or blur)
  const lineInput = document.getElementById("line-input");
  const applyLine = () => {
    const val = parseFloat(lineInput.value);
    state.modalLine = isNaN(val) ? null : val;
    loadPlayerHistory();
  };
  lineInput.addEventListener("keydown", e => { if (e.key === "Enter") applyLine(); });
  lineInput.addEventListener("change", applyLine);

  // Close player modal
  document.getElementById("modal-close-btn").addEventListener("click", closeModal);
  document.getElementById("modal-backdrop").addEventListener("click", closeModal);

  // Close game panel
  document.getElementById("game-backdrop").addEventListener("click", closeGamePanel);

  // Refresh button
  document.getElementById("refresh-btn").addEventListener("click", async () => {
    const btn = document.getElementById("refresh-btn");
    btn.classList.add("loading");
    await fetch("/api/refresh");
    state.players = [];
    state.props = {};
    await Promise.all([loadGames(), loadPlayers()]);
    btn.classList.remove("loading");
  });
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupEvents();
  loadGames();
  loadPlayers();
});
