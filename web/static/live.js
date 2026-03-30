/* NBA Predictor — Live Tracker */

const LS_KEY     = "nba_live_parlays";
const LS_OLD_KEY = "nba_live_picks";

// ── State ─────────────────────────────────────────────────────────────────
// parlays: [{ id, name, legs: [{ id, player, stat, line, odds }] }]
let parlays        = [];
let games          = [];
let boxscores      = {};
let allHeadshots   = {};   // name.lower() → url (from /api/live/headshots)
let activeStat      = "pts";
let activeDirection = "over";
let activePickType   = "prop";   // "prop" | "ml" | "spread" | "total"
let activeMLTeam     = null;     // { abbr, eventId, isHome }
let activeSpreadTeam = null;     // { abbr, eventId, isHome }
let activeTotalDir   = "over";
let refreshTimer   = null;
let allPlayers     = [];
let suggFocusIdx   = -1;
let dragLegId      = null;
let dragFromGroup  = null;

// ── Player autocomplete ───────────────────────────────────────────────────
async function fetchPlayerList() {
  try {
    const res = await fetch("/api/live/players");
    allPlayers = await res.json();
  } catch { allPlayers = []; }
}

const playerInput = document.getElementById("pick-player-input");
const suggestionsEl = document.getElementById("player-suggestions");

function showSuggestions(query) {
  const q = query.trim().toLowerCase();
  if (!q || q.length < 2) { hideSuggestions(); return; }

  const matches = allPlayers
    .filter(n => n.toLowerCase().includes(q))
    .slice(0, 8);

  if (!matches.length) { hideSuggestions(); return; }

  suggFocusIdx = -1;
  suggestionsEl.innerHTML = matches.map((name, i) => {
    const highlighted = name.replace(
      new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"),
      m => `<mark>${m}</mark>`
    );
    return `<div class="suggestion-item" data-idx="${i}" data-name="${name}">${highlighted}</div>`;
  }).join("");

  suggestionsEl.classList.add("visible");

  suggestionsEl.querySelectorAll(".suggestion-item").forEach(item => {
    item.addEventListener("mousedown", e => {
      e.preventDefault();
      selectSuggestion(item.dataset.name);
    });
  });
}

function hideSuggestions() {
  suggestionsEl.classList.remove("visible");
  suggestionsEl.innerHTML = "";
  suggFocusIdx = -1;
}

function selectSuggestion(name) {
  playerInput.value = name;
  hideSuggestions();
  document.getElementById("pick-line-input").focus();
}

function moveFocus(dir) {
  const items = suggestionsEl.querySelectorAll(".suggestion-item");
  if (!items.length) return;
  items[suggFocusIdx]?.classList.remove("focused");
  suggFocusIdx = Math.max(0, Math.min(items.length - 1, suggFocusIdx + dir));
  items[suggFocusIdx].classList.add("focused");
  items[suggFocusIdx].scrollIntoView({ block: "nearest" });
}

playerInput.addEventListener("input", () => showSuggestions(playerInput.value));
playerInput.addEventListener("blur", () => setTimeout(hideSuggestions, 150));
playerInput.addEventListener("keydown", e => {
  if (!suggestionsEl.classList.contains("visible")) return;
  if (e.key === "ArrowDown")  { e.preventDefault(); moveFocus(1); }
  if (e.key === "ArrowUp")    { e.preventDefault(); moveFocus(-1); }
  if (e.key === "Enter") {
    const focused = suggestionsEl.querySelector(".suggestion-item.focused");
    if (focused) { e.preventDefault(); selectSuggestion(focused.dataset.name); }
  }
  if (e.key === "Escape") hideSuggestions();
});

// ── Name normalization (strips diacritics: Jokić → Jokic) ────────────────
function normName(name) {
  return name.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

// ── Headshots ─────────────────────────────────────────────────────────────
async function fetchAllHeadshots() {
  try {
    const res = await fetch("/api/live/headshots");
    allHeadshots = await res.json();
  } catch { allHeadshots = {}; }
}

function getHeadshot(name) {
  return allHeadshots[normName(name)] || "";
}

// ── Persistence ───────────────────────────────────────────────────────────
function loadParlays() {
  try {
    const saved = JSON.parse(localStorage.getItem(LS_KEY));
    if (saved?.length) { parlays = saved; return; }
  } catch {}
  // Migrate old picks format
  try {
    const old = JSON.parse(localStorage.getItem(LS_OLD_KEY));
    if (old?.length) {
      parlays = [{ id: Date.now(), name: "Parlay 1", legs: old }];
      saveParlays();
      localStorage.removeItem(LS_OLD_KEY);
      return;
    }
  } catch {}
  parlays = [{ id: Date.now(), name: "Parlay 1", legs: [] }];
}

function saveParlays() {
  localStorage.setItem(LS_KEY, JSON.stringify(parlays));
}

// ── Fetch helpers ─────────────────────────────────────────────────────────
async function fetchGames() {
  try {
    const res = await fetch("/api/live/games");
    games = await res.json();
  } catch { games = []; }
}

async function fetchBoxscore(eventId) {
  try {
    const res = await fetch(`/api/live/boxscore/${eventId}`);
    boxscores[eventId] = await res.json();
  } catch { /* keep stale */ }
}

async function refreshAll() {
  await fetchGames();
  // Fetch boxscores for all games (status string can be unreliable)
  await Promise.all(games.map(g => fetchBoxscore(g.event_id)));
  // Render everything after both games + boxscores are ready
  renderGamesStrip();
  renderParlays();
  renderBoxscoreSidebar();
}

// ── Game strip ────────────────────────────────────────────────────────────
function renderGamesStrip() {
  const el = document.getElementById("live-games-strip");
  if (!games.length) {
    el.innerHTML = `<div class="state-empty" style="padding:20px 0;color:var(--text-muted);font-size:13px">No games today.</div>`;
    return;
  }

  el.innerHTML = games.map(g => {
    const isLive  = g.status === "STATUS_IN_PROGRESS";
    const isFinal = g.status === "STATUS_FINAL";

    let statusHtml;
    if (isLive) {
      const bs = Object.values(boxscores).find(b => b.event_id === g.event_id);
      const clock = bs ? `Q${bs.period} ${bs.clock}` : "LIVE";
      statusHtml = `<span class="live-badge">● ${clock}</span>`;
    } else if (isFinal) {
      statusHtml = `<span class="live-badge final">FINAL</span>`;
    } else {
      const t = new Date(g.game_time);
      const timeStr = t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      statusHtml = `<span class="live-chip-time">${timeStr}</span>`;
    }

    const awayScore = (isLive || isFinal) ? `<span class="live-chip-score">${g.away_score}</span>` : "";
    const homeScore = (isLive || isFinal) ? `<span class="live-chip-score">${g.home_score}</span>` : "";

    return `
      <div class="live-game-chip${isLive ? " is-live" : ""}${isFinal ? " is-final" : ""}">
        <div class="live-chip-team">
          ${g.away_logo ? `<img src="${g.away_logo}" class="live-chip-logo" alt="">` : ""}
          <span class="live-chip-abbr">${g.away_team}</span>
          ${awayScore}
        </div>
        <div class="live-chip-middle">
          ${statusHtml}
        </div>
        <div class="live-chip-team right">
          ${homeScore}
          <span class="live-chip-abbr">${g.home_team}</span>
          ${g.home_logo ? `<img src="${g.home_logo}" class="live-chip-logo" alt="">` : ""}
        </div>
      </div>`;
  }).join("");
}

// ── Projection ────────────────────────────────────────────────────────────
function computeProjection(playerData, eventId, stat) {
  const current   = playerData[stat] ?? 0;
  const minPlayed = playerData.min   || 0;
  const avgMin    = playerData.avg_min || 32;

  if (minPlayed < 0.5) return null;

  const rate = current / minPlayed;

  // Estimate minutes left: how many minutes remain in the game for this player.
  // Use the game clock when available, otherwise fall back to avg_min pace.
  const bs = boxscores[eventId];
  let gameRemaining = Math.max(0, avgMin - minPlayed); // sensible default

  if (bs && !bs.is_final) {
    const period = bs.period || 1;
    const clock  = bs.clock  || "";
    let clockMin = 12;
    if (clock && clock !== "0:00") {
      const parts  = clock.split(":");
      const parsed = parseInt(parts[0]) + parseInt(parts[1] || "0") / 60;
      if (!isNaN(parsed)) clockMin = parsed;
    } else {
      clockMin = 0;
    }
    if (period <= 4) {
      const elapsed = (period - 1) * 12 + (12 - clockMin);
      gameRemaining = Math.max(0, 48 - elapsed);
    }
  }

  const playerRemaining = Math.max(0, avgMin - minPlayed);
  const minutesLeft = Math.min(playerRemaining, gameRemaining);
  const projected   = Math.round(current + rate * minutesLeft);
  return isNaN(projected) ? null : projected;
}

// ── Find player in boxscores ──────────────────────────────────────────────
function findPlayerStats(playerName) {
  const needle = normName(playerName);
  for (const [eventId, bs] of Object.entries(boxscores)) {
    for (const [name, stats] of Object.entries(bs.players || {})) {
      if (normName(name) === needle) {
        return { stats, eventId, bs };
      }
    }
  }
  return null;
}

// ── Status badge ──────────────────────────────────────────────────────────
function pickStatus(current, projected, line, isFinal, isLive, found, minPlayed, direction) {
  const dir = direction || "over";
  if (!found) return { label: "PENDING", cls: "pending" };

  if (dir === "under") {
    // Under: busted as soon as current exceeds line (points can't decrease)
    if (current > line)  return { label: "MISS ✗",  cls: "miss" };
    if (isFinal)         return { label: "HIT ✓",   cls: "hit" };
    const gameStarted = isLive || (minPlayed > 0);
    if (!gameStarted)    return { label: "NOT YET", cls: "pending" };
    if (projected === null) return { label: "LIVE", cls: "live" };
    return projected <= line
      ? { label: "ON PACE", cls: "on-pace" }
      : { label: "BEHIND",  cls: "behind" };
  }

  // Over (default)
  if (current >= line)  return { label: "HIT ✓",   cls: "hit" };
  if (isFinal)          return { label: "MISS ✗",  cls: "miss" };
  const gameStarted = isLive || (minPlayed > 0);
  if (!gameStarted)     return { label: "NOT YET", cls: "pending" };
  if (projected === null) return { label: "LIVE",  cls: "live" };
  return projected >= line
    ? { label: "ON PACE", cls: "on-pace" }
    : { label: "BEHIND",  cls: "behind" };
}

// ── ML status ─────────────────────────────────────────────────────────────
function mlStatus(isHome, bs) {
  if (!bs) return { label: "PENDING", cls: "pending" };
  const isFinal = bs.is_final;
  const isLive  = bs.is_live;

  // Determine winner/loser from scores for final games
  if (isFinal) {
    const scores = bs.scores || {};
    const vals = Object.entries(scores);
    if (vals.length >= 2) {
      // Find home/away team abbrs from the game stored on the leg (resolved at render time)
      // Use home_win_prob > 50 as proxy for home team winning when final
      const homeWon = vals.reduce((best, v) => parseInt(v[1]) > parseInt(best[1]) ? v : best, vals[0]);
      // We'll resolve this by checking scores directly in renderParlays
      return { label: "FINAL", cls: "pending" };
    }
    return { label: "FINAL", cls: "pending" };
  }

  if (!isLive) return { label: "NOT YET", cls: "pending" };

  const prob = bs.home_win_prob;
  if (prob === null || prob === undefined) return { label: "LIVE", cls: "live" };

  const myProb = isHome ? prob : (100 - prob);
  if (myProb >= 60) return { label: "ON PACE", cls: "on-pace" };
  if (myProb >= 45) return { label: "CLOSE",   cls: "live" };
  return { label: "BEHIND", cls: "behind" };
}

// ── Total projection + status ──────────────────────────────────────────────
function computeTotalProjection(bs) {
  if (!bs || !bs.is_live) return null;
  const scores = bs.scores || {};
  const vals = Object.values(scores).map(v => parseInt(v) || 0);
  if (vals.length < 2) return null;
  const currentTotal = vals.reduce((a, b) => a + b, 0);

  const period   = bs.period || 1;
  const clock    = bs.clock  || "";
  let clockMin   = 12;
  if (clock && clock !== "0:00") {
    const parts = clock.split(":");
    const parsed = parseInt(parts[0]) + parseInt(parts[1] || "0") / 60;
    if (!isNaN(parsed)) clockMin = parsed;
  } else {
    clockMin = 0;
  }
  if (period > 4) return currentTotal; // OT — use current as projection
  const elapsedMin   = (period - 1) * 12 + (12 - clockMin);
  if (elapsedMin < 1) return null;
  const remainingMin = Math.max(0, 48 - elapsedMin);
  const pace         = currentTotal / elapsedMin;
  return Math.round(currentTotal + pace * remainingMin);
}

function totalStatus(line, direction, bs) {
  const dir = direction || "over";
  if (!bs) return { label: "PENDING", cls: "pending" };

  const scores = bs.scores || {};
  const vals = Object.values(scores).map(v => parseInt(v) || 0);
  const currentTotal = vals.reduce((a, b) => a + b, 0);

  if (bs.is_final) {
    if (dir === "over")  return currentTotal > line  ? { label: "HIT ✓", cls: "hit" } : { label: "MISS ✗", cls: "miss" };
    return currentTotal < line ? { label: "HIT ✓", cls: "hit" } : { label: "MISS ✗", cls: "miss" };
  }

  if (!bs.is_live) return { label: "NOT YET", cls: "pending" };

  // Under: bust immediately if already over the line
  if (dir === "under" && currentTotal >= line) return { label: "MISS ✗", cls: "miss" };

  const projected = computeTotalProjection(bs);
  if (projected === null) return { label: "LIVE", cls: "live" };

  if (dir === "over")  return projected >= line ? { label: "ON PACE", cls: "on-pace" } : { label: "BEHIND", cls: "behind" };
  return projected <= line ? { label: "ON PACE", cls: "on-pace" } : { label: "BEHIND", cls: "behind" };
}

// ── Render a single leg (all types) ───────────────────────────────────────
function renderLeg(leg, group) {
  if (leg.type === "ml")     return renderMLLeg(leg, group);
  if (leg.type === "spread") return renderSpreadLeg(leg, group);
  if (leg.type === "total")  return renderTotalLeg(leg, group);
  return renderPropLeg(leg, group);
}

function renderPropLeg(leg, group) {
  const statLabel = { pts: "PTS", reb: "REB", ast: "AST", tpm: "3PM" };
  const found     = findPlayerStats(leg.player);
  const stats     = found?.stats;
  const eventId   = found?.eventId;
  const bs        = found?.bs;
  const dir       = leg.direction || "over";
  const current   = stats ? (stats[leg.stat] ?? 0) : null;
  const minPlayed = stats?.min || 0;
  const isLive    = bs?.is_live  ?? false;
  const isFinal   = bs?.is_final ?? false;
  const projected = (stats && eventId) ? computeProjection(stats, eventId, leg.stat) : null;
  const pct       = current !== null ? Math.min(100, Math.round((current / leg.line) * 100)) : 0;
  const status    = pickStatus(current, projected, leg.line, isFinal, isLive, !!found, minPlayed, dir);

  const barColor = dir === "under"
    ? (status.cls === "hit" || status.cls === "on-pace" ? "var(--green)" : status.cls === "miss" || status.cls === "behind" ? "var(--red)" : "var(--accent)")
    : (status.cls === "hit" ? "var(--green)" : status.cls === "on-pace" ? "var(--yellow)" : status.cls === "miss" || status.cls === "behind" ? "var(--red)" : "var(--accent)");

  const headshot = stats?.headshot || getHeadshot(leg.player);
  const projTxt  = projected !== null && (isLive || minPlayed > 0) && !isFinal
    ? `<span class="leg-proj">~${projected}</span>` : "";
  const dirPill  = `<span class="pick-dir-pill ${dir}">${dir === "over" ? "O" : "U"}</span>`;

  return `
    <div class="parlay-leg ${status.cls}" draggable="true" data-leg-id="${leg.id}" data-group-id="${group.id}">
      <div class="leg-drag-handle">⠿</div>
      ${headshot ? `<img src="${headshot}" class="leg-headshot" alt="">` : `<div class="leg-headshot-ph"></div>`}
      <div class="leg-info">
        <div class="leg-name">
          ${leg.player.split(" ").slice(-1)[0]}
          <span class="pick-stat-pill">${statLabel[leg.stat]}</span>
          ${dirPill}
        </div>
        <div class="leg-progress-row">
          <span class="leg-score">${current !== null ? current : "—"}/${leg.line}</span>
          ${projTxt}
          <div class="leg-bar-wrap"><div class="leg-bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
        </div>
      </div>
      <span class="pick-status-badge ${status.cls}">${status.label}</span>
      <button class="pick-remove-btn" onclick="removeLeg(${leg.id},${group.id})">✕</button>
    </div>`;
}

function renderMLLeg(leg, group) {
  const bs      = boxscores[leg.eventId];
  const game    = games.find(g => g.event_id === leg.eventId);
  const scores  = bs?.scores || {};
  const isFinal = bs?.is_final ?? false;
  const isLive  = bs?.is_live  ?? false;

  // Determine hit/miss when final
  let status;
  if (isFinal && game) {
    const myScore   = parseInt(scores[leg.team]) || 0;
    const oppTeam   = leg.isHome ? game.away_team : game.home_team;
    const oppScore  = parseInt(scores[oppTeam])  || 0;
    status = myScore > oppScore ? { label: "HIT ✓", cls: "hit" } : { label: "MISS ✗", cls: "miss" };
  } else {
    status = mlStatus(leg.isHome, bs);
  }

  const prob     = bs?.home_win_prob;
  const myProb   = prob !== null && prob !== undefined ? (leg.isHome ? prob : 100 - prob) : null;
  const pct      = myProb !== null ? Math.round(myProb) : 50;
  const barColor = status.cls === "hit" ? "var(--green)" : status.cls === "on-pace" ? "var(--yellow)" :
                   status.cls === "miss" || status.cls === "behind" ? "var(--red)" : "var(--accent)";
  const probTxt  = myProb !== null && (isLive || isFinal) ? `<span class="leg-proj">${Math.round(myProb)}%</span>` : "";
  const logo     = game ? (leg.isHome ? game.home_logo : game.away_logo) : "";

  return `
    <div class="parlay-leg ${status.cls}" draggable="true" data-leg-id="${leg.id}" data-group-id="${group.id}">
      <div class="leg-drag-handle">⠿</div>
      ${logo ? `<img src="${logo}" class="leg-headshot" alt="" style="border-radius:4px">` : `<div class="leg-headshot-ph"></div>`}
      <div class="leg-info">
        <div class="leg-name">
          ${leg.team}
          <span class="pick-stat-pill">ML</span>
        </div>
        <div class="leg-progress-row">
          <span class="leg-score">${isLive || isFinal ? `${pct}% win prob` : leg.gameLabel}</span>
          ${probTxt}
          <div class="leg-bar-wrap"><div class="leg-bar-fill" style="width:${Math.min(100,pct)}%;background:${barColor}"></div></div>
        </div>
      </div>
      <span class="pick-status-badge ${status.cls}">${status.label}</span>
      <button class="pick-remove-btn" onclick="removeLeg(${leg.id},${group.id})">✕</button>
    </div>`;
}

function renderTotalLeg(leg, group) {
  const bs      = boxscores[leg.eventId];
  const game    = games.find(g => g.event_id === leg.eventId);
  const scores  = bs?.scores || {};
  const vals    = Object.values(scores).map(v => parseInt(v) || 0);
  const current = vals.reduce((a, b) => a + b, 0);
  const isLive  = bs?.is_live  ?? false;
  const isFinal = bs?.is_final ?? false;
  const dir     = leg.direction || "over";
  const status  = totalStatus(leg.line, dir, bs);
  const projected = isLive ? computeTotalProjection(bs) : null;

  const pct      = leg.line > 0 ? Math.min(100, Math.round((current / leg.line) * 100)) : 0;
  const barColor = status.cls === "hit" ? "var(--green)" : status.cls === "on-pace" ? "var(--yellow)" :
                   status.cls === "miss" || status.cls === "behind" ? "var(--red)" : "var(--accent)";
  const projTxt  = projected !== null ? `<span class="leg-proj">~${projected}</span>` : "";
  const dirPill  = `<span class="pick-dir-pill ${dir}">${dir === "over" ? "O" : "U"}</span>`;
  const scoreStr = (isLive || isFinal) ? `${current}/${leg.line}` : leg.gameLabel;

  return `
    <div class="parlay-leg ${status.cls}" draggable="true" data-leg-id="${leg.id}" data-group-id="${group.id}">
      <div class="leg-drag-handle">⠿</div>
      <div class="leg-headshot-ph" style="font-size:18px;display:flex;align-items:center;justify-content:center">⛹</div>
      <div class="leg-info">
        <div class="leg-name">
          ${leg.gameLabel}
          <span class="pick-stat-pill">TOT</span>
          ${dirPill}
        </div>
        <div class="leg-progress-row">
          <span class="leg-score">${scoreStr}</span>
          ${projTxt}
          <div class="leg-bar-wrap"><div class="leg-bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
        </div>
      </div>
      <span class="pick-status-badge ${status.cls}">${status.label}</span>
      <button class="pick-remove-btn" onclick="removeLeg(${leg.id},${group.id})">✕</button>
    </div>`;
}

// ── Spread projection + status ────────────────────────────────────────────
function computeMarginProjection(bs, isHome) {
  if (!bs || !bs.is_live) return null;
  const scores = bs.scores || {};
  const vals   = Object.entries(scores).map(([k, v]) => ({ abbr: k, pts: parseInt(v) || 0 }));
  if (vals.length < 2) return null;

  const period   = bs.period || 1;
  const clock    = bs.clock  || "";
  let clockMin   = 12;
  if (clock && clock !== "0:00") {
    const parts = clock.split(":");
    const parsed = parseInt(parts[0]) + parseInt(parts[1] || "0") / 60;
    if (!isNaN(parsed)) clockMin = parsed;
  } else {
    clockMin = 0;
  }
  if (period > 4) return null;
  const elapsed    = (period - 1) * 12 + (12 - clockMin);
  const remaining  = Math.max(0, 48 - elapsed);
  if (elapsed < 1) return null;

  const total      = vals.reduce((s, v) => s + v.pts, 0);
  const pace       = total / elapsed;            // total pts/min
  const projTotal  = total + pace * remaining;   // project total final score
  // Spread only cares about margin, not total — project each team proportionally
  const home = vals.find(v => v.abbr === Object.keys(scores)[1]) || vals[1];
  const away = vals.find(v => v.abbr !== home?.abbr) || vals[0];
  const currentMargin = (isHome ? home.pts - away.pts : away.pts - home.pts);
  const marginRate    = currentMargin / elapsed;
  return Math.round(currentMargin + marginRate * remaining);
}

function spreadStatus(spread, isHome, bs, game) {
  if (!bs) return { label: "PENDING", cls: "pending" };
  const scores   = bs.scores || {};
  const homeAbbr = game?.home_team;
  const awayAbbr = game?.away_team;
  const homeScore = parseInt(scores[homeAbbr]) || 0;
  const awayScore = parseInt(scores[awayAbbr]) || 0;
  const margin    = isHome ? homeScore - awayScore : awayScore - homeScore;
  // Covers if margin > spread (e.g. spread=-6.5, margin=7 → covers)
  const covers    = margin > -spread;

  if (bs.is_final) return covers ? { label: "HIT ✓", cls: "hit" } : { label: "MISS ✗", cls: "miss" };
  if (!bs.is_live) return { label: "NOT YET", cls: "pending" };

  const projected = computeMarginProjection(bs, isHome);
  if (projected === null) return { label: "LIVE", cls: "live" };
  const projCovers = projected > -spread;
  return projCovers ? { label: "ON PACE", cls: "on-pace" } : { label: "BEHIND", cls: "behind" };
}

function renderSpreadLeg(leg, group) {
  const bs      = boxscores[leg.eventId];
  const game    = games.find(g => g.event_id === leg.eventId);
  const scores  = bs?.scores || {};
  const homeAbbr = game?.home_team;
  const awayAbbr = game?.away_team;
  const homeScore = parseInt(scores[homeAbbr]) || 0;
  const awayScore = parseInt(scores[awayAbbr]) || 0;
  const margin    = leg.isHome ? homeScore - awayScore : awayScore - homeScore;
  const isLive    = bs?.is_live  ?? false;
  const isFinal   = bs?.is_final ?? false;
  const status    = spreadStatus(leg.spread, leg.isHome, bs, game);
  const projected = isLive ? computeMarginProjection(bs, leg.isHome) : null;

  const spreadLabel = leg.spread >= 0 ? `+${leg.spread}` : `${leg.spread}`;
  // Progress bar: 0 at spread-10, full at spread+10 (centered on the spread)
  const center = leg.spread;
  const pct    = (isLive || isFinal) ? Math.min(100, Math.max(0, Math.round(((margin - center + 10) / 20) * 100))) : 50;
  const barColor = status.cls === "hit" ? "var(--green)" : status.cls === "on-pace" ? "var(--yellow)" :
                   status.cls === "miss" || status.cls === "behind" ? "var(--red)" : "var(--accent)";
  const projTxt  = projected !== null ? `<span class="leg-proj">~${projected > 0 ? "+" : ""}${projected}</span>` : "";
  const logo     = game ? (leg.isHome ? game.home_logo : game.away_logo) : "";
  const scoreStr = (isLive || isFinal) ? `${margin > 0 ? "+" : ""}${margin} (need ${spreadLabel})` : leg.gameLabel;

  return `
    <div class="parlay-leg ${status.cls}" draggable="true" data-leg-id="${leg.id}" data-group-id="${group.id}">
      <div class="leg-drag-handle">⠿</div>
      ${logo ? `<img src="${logo}" class="leg-headshot" alt="" style="border-radius:4px">` : `<div class="leg-headshot-ph"></div>`}
      <div class="leg-info">
        <div class="leg-name">
          ${leg.team} <span class="pick-stat-pill">ATS</span>
          <span class="pick-dir-pill over">${spreadLabel}</span>
        </div>
        <div class="leg-progress-row">
          <span class="leg-score">${scoreStr}</span>
          ${projTxt}
          <div class="leg-bar-wrap"><div class="leg-bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
        </div>
      </div>
      <span class="pick-status-badge ${status.cls}">${status.label}</span>
      <button class="pick-remove-btn" onclick="removeLeg(${leg.id},${group.id})">✕</button>
    </div>`;
}

// ── Render all parlay groups ──────────────────────────────────────────────
function renderPicks() { renderParlays(); } // alias kept for refreshAll calls

function renderParlays() {
  const container = document.getElementById("parlays-container");

  const allLegs = parlays.flatMap(g => g.legs);
  if (!allLegs.length) {
    container.innerHTML = `
      <div class="picks-empty">
        <div class="picks-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
          </svg>
        </div>
        <p>No picks yet.</p>
        <button class="btn-add-pick-empty" onclick="document.getElementById('add-pick-btn').click()">Add your first pick</button>
      </div>`;
    return;
  }

  container.innerHTML = parlays.map(group => {
    const legsHtml = group.legs.map(leg => renderLeg(leg, group)).join("");

    const canDelete = parlays.length > 1;
    return `
      <div class="parlay-group-card" data-group-id="${group.id}">
        <div class="parlay-group-header">
          <span class="parlay-group-name">${group.name}</span>
          <span class="parlay-leg-count">${group.legs.length} leg${group.legs.length !== 1 ? "s" : ""}</span>
          ${canDelete ? `<button class="parlay-group-del" onclick="removeGroup(${group.id})">✕</button>` : ""}
        </div>
        <div class="parlay-legs-list" data-group-id="${group.id}">
          ${legsHtml}
          <div class="leg-drop-zone" data-group-id="${group.id}">Drop pick here</div>
        </div>
      </div>`;
  }).join("");

  attachDragDrop();
}

function attachDragDrop() {
  document.querySelectorAll(".parlay-leg[draggable]").forEach(el => {
    el.addEventListener("dragstart", e => {
      dragLegId    = parseInt(el.dataset.legId);
      dragFromGroup = parseInt(el.dataset.groupId);
      el.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });
    el.addEventListener("dragend", () => {
      el.classList.remove("dragging");
      document.querySelectorAll(".leg-drop-zone").forEach(z => z.classList.remove("active"));
    });
  });
  document.querySelectorAll(".leg-drop-zone").forEach(zone => {
    const toGroupId = parseInt(zone.dataset.groupId);
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("active"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("active"));
    zone.addEventListener("drop", e => {
      e.preventDefault();
      zone.classList.remove("active");
      if (dragLegId !== null) moveLeg(dragLegId, dragFromGroup, toGroupId);
      dragLegId = dragFromGroup = null;
    });
  });
}

// ── OLD renderPicks stub (kept for line count alignment) ──────────────────
function _oldRenderPicks_REMOVED() {
  const grid  = document.getElementById("picks-grid");
  const empty = document.getElementById("picks-empty");
  const countEl = document.getElementById("picks-count");

  countEl.textContent = picks.length;

  if (!picks.length) {
    grid.style.display  = "none";
    empty.style.display = "flex";
    return;
  }
  grid.style.display  = "grid";
  empty.style.display = "none";

  const statLabel = { pts: "PTS", reb: "REB", ast: "AST", tpm: "3PM" };

  grid.innerHTML = picks.map(pick => {
    const found = findPlayerStats(pick.player);
    const stats   = found?.stats;
    const eventId = found?.eventId;
    const bs      = found?.bs;

    const current  = stats ? (stats[pick.stat] ?? 0) : null;
    const minPlayed = stats?.min || 0;
    const minStr   = stats?.min_str || "—";
    const headshot = stats?.headshot || "";
    const team     = stats?.team || "";

    const isLive  = bs?.is_live  ?? false;
    const isFinal = bs?.is_final ?? false;

    const projected = (stats && eventId)
      ? computeProjection(stats, eventId, pick.stat)
      : null;

    const pct = current !== null
      ? Math.min(100, Math.round((current / pick.line) * 100))
      : 0;

    const status = pickStatus(current, projected, pick.line, isFinal, isLive, !!found, minPlayed);

    const progressColor =
      status.cls === "hit"     ? "var(--green)"  :
      status.cls === "on-pace" ? "var(--yellow)" :
      status.cls === "behind"  ? "var(--red)"    :
      status.cls === "miss"    ? "var(--red)"    :
      "var(--accent)";

    const oddsDisplay = pick.odds
      ? `<span class="pick-odds-tag">${pick.odds > 0 ? "+" : ""}${pick.odds}</span>`
      : "";

    const projDisplay = projected !== null && isLive
      ? `<div class="pick-proj">~${projected} proj</div>`
      : "";

    const minDisplay = minStr !== "—" && minStr !== "0:00"
      ? `<div class="pick-mins">${minStr} min played</div>`
      : "";

    return `
      <div class="pick-tracker-card ${status.cls}">
        <div class="pick-tracker-top">
          <div class="pick-tracker-player">
            ${headshot
              ? `<img src="${headshot}" class="pick-tracker-headshot" alt="">`
              : `<div class="pick-tracker-headshot-placeholder"></div>`}
            <div class="pick-tracker-info">
              <div class="pick-tracker-name">${pick.player}</div>
              <div class="pick-tracker-meta">
                ${team ? `<span>${team}</span>` : ""}
                <span class="pick-stat-pill">${statLabel[pick.stat]}</span>
                ${oddsDisplay}
              </div>
            </div>
          </div>
          <div class="pick-tracker-right">
            <span class="pick-status-badge ${status.cls}">${status.label}</span>
            <button class="pick-remove-btn" onclick="removePick(${pick.id})" title="Remove">✕</button>
          </div>
        </div>

        <div class="pick-tracker-stats">
          <div class="pick-stat-display">
            <span class="pick-current">${current !== null ? current : "—"}</span>
            <span class="pick-line-sep">/</span>
            <span class="pick-line-val">${pick.line}</span>
          </div>
          ${projDisplay}
          ${minDisplay}
        </div>

        <div class="pick-progress-track">
          <div class="pick-progress-fill" style="width:${pct}%;background:${progressColor}"></div>
        </div>
        <div class="pick-progress-labels">
          <span>0</span>
          <span style="font-weight:600;color:var(--text)">${pick.line}</span>
        </div>
      </div>`;
  }).join("");
}


// ── Parlay group management ───────────────────────────────────────────────
function addGroup() {
  parlays.push({ id: Date.now(), name: `Parlay ${parlays.length + 1}`, legs: [] });
  saveParlays();
  renderParlays();
}

function removeGroup(groupId) {
  parlays = parlays.filter(g => g.id !== groupId);
  if (!parlays.length) parlays = [{ id: Date.now(), name: "Parlay 1", legs: [] }];
  saveParlays();
  renderParlays();
}

function removeLeg(legId, groupId) {
  const g = parlays.find(g => g.id === groupId);
  if (g) g.legs = g.legs.filter(l => l.id !== legId);
  saveParlays();
  renderParlays();
}

function moveLeg(legId, fromGroupId, toGroupId) {
  if (fromGroupId === toGroupId) return;
  const from = parlays.find(g => g.id === fromGroupId);
  const to   = parlays.find(g => g.id === toGroupId);
  if (!from || !to) return;
  const leg = from.legs.find(l => l.id === legId);
  if (!leg) return;
  from.legs = from.legs.filter(l => l.id !== legId);
  to.legs.push(leg);
  saveParlays();
  renderParlays();
}

// ── Add Pick Modal ────────────────────────────────────────────────────────
function _gameLabel(g) {
  return `${g.away_team} @ ${g.home_team}`;
}

function _populateGameSelect(selId, onChange) {
  const sel = document.getElementById(selId);
  if (!games.length) {
    sel.innerHTML = `<option value="">No games today</option>`;
    return;
  }
  sel.innerHTML = games.map(g =>
    `<option value="${g.event_id}">${_gameLabel(g)}</option>`
  ).join("");
  if (onChange) { sel.removeEventListener("change", sel._onChange || (() => {})); sel._onChange = onChange; sel.addEventListener("change", onChange); onChange(); }
}

function _buildTeamTabs(tabsEl, game, onSelect) {
  if (!game) { tabsEl.innerHTML = ""; return; }
  const opts = [
    { abbr: game.away_team, isHome: false },
    { abbr: game.home_team, isHome: true  },
  ];
  tabsEl.innerHTML = opts.map((t, i) =>
    `<button class="pick-dir-tab${i === 0 ? " active" : ""}" data-abbr="${t.abbr}" data-home="${t.isHome}" data-event="${game.event_id}">${t.abbr}</button>`
  ).join("");
  onSelect({ abbr: opts[0].abbr, isHome: false, eventId: game.event_id });
  tabsEl.querySelectorAll(".pick-dir-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      tabsEl.querySelectorAll(".pick-dir-tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      onSelect({ abbr: btn.dataset.abbr, isHome: btn.dataset.home === "true", eventId: btn.dataset.event });
    });
  });
}

function _updateMLTeamTabs() {
  const sel   = document.getElementById("ml-game-select");
  const game  = games.find(g => g.event_id === sel.value);
  _buildTeamTabs(document.getElementById("ml-team-tabs"), game, t => { activeMLTeam = t; });
}

function _updateSpreadTeamTabs() {
  const sel  = document.getElementById("spread-game-select");
  const game = games.find(g => g.event_id === sel.value);
  _buildTeamTabs(document.getElementById("spread-team-tabs"), game, t => { activeSpreadTeam = t; });
}

function _switchPickType(type) {
  activePickType = type;
  document.querySelectorAll(".pick-type-tab").forEach(t =>
    t.classList.toggle("active", t.dataset.type === type));
  document.getElementById("section-prop").style.display   = type === "prop"   ? "" : "none";
  document.getElementById("section-ml").style.display     = type === "ml"     ? "" : "none";
  document.getElementById("section-spread").style.display = type === "spread" ? "" : "none";
  document.getElementById("section-total").style.display  = type === "total"  ? "" : "none";
}

function openAddPick() {
  // Populate group selector
  const sel = document.getElementById("pick-group-select");
  sel.innerHTML = parlays.map(g => `<option value="${g.id}">${g.name}</option>`).join("");
  document.getElementById("group-select-wrap").style.display = parlays.length > 1 ? "" : "none";

  // Populate game dropdowns
  _populateGameSelect("ml-game-select",     _updateMLTeamTabs);
  _populateGameSelect("spread-game-select", _updateSpreadTeamTabs);
  _populateGameSelect("total-game-select",  null);

  _switchPickType("prop");
  document.getElementById("add-pick-backdrop").classList.add("visible");
  document.getElementById("add-pick-modal").classList.add("visible");
  document.getElementById("pick-player-input").focus();
}

function closeAddPick() {
  document.getElementById("add-pick-backdrop").classList.remove("visible");
  document.getElementById("add-pick-modal").classList.remove("visible");
  document.getElementById("pick-player-input").value = "";
  document.getElementById("pick-line-input").value = "";
  document.getElementById("total-line-input").value = "";
  document.getElementById("spread-line-input").value = "";
  activeStat = "pts";
  activeDirection = "over";
  activeTotalDir = "over";
  activePickType = "prop";
  activeMLTeam = null;
  activeSpreadTeam = null;
  document.querySelectorAll(".pick-stat-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.stat === "pts");
  });
  document.querySelectorAll(".pick-dir-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.dir === "over");
  });
}


function confirmPick() {
  const groupId = parseInt(document.getElementById("pick-group-select").value) || parlays[0]?.id;
  const group   = parlays.find(g => g.id === groupId) || parlays[0];
  if (!group) return;

  if (activePickType === "prop") {
    const player = document.getElementById("pick-player-input").value.trim();
    const line   = parseFloat(document.getElementById("pick-line-input").value);
    if (!player || isNaN(line)) { document.getElementById("pick-player-input").focus(); return; }
    group.legs.push({ id: Date.now(), type: "prop", player, stat: activeStat, line, direction: activeDirection });

  } else if (activePickType === "ml") {
    if (!activeMLTeam) return;
    const game = games.find(g => g.event_id === activeMLTeam.eventId);
    if (!game) return;
    const isHome = activeMLTeam.abbr === game.home_team;
    group.legs.push({ id: Date.now(), type: "ml", team: activeMLTeam.abbr, eventId: activeMLTeam.eventId,
      gameLabel: _gameLabel(game), isHome });

  } else if (activePickType === "spread") {
    if (!activeSpreadTeam) return;
    const spread = parseFloat(document.getElementById("spread-line-input").value);
    if (isNaN(spread)) { document.getElementById("spread-line-input").focus(); return; }
    const game = games.find(g => g.event_id === activeSpreadTeam.eventId);
    group.legs.push({ id: Date.now(), type: "spread", team: activeSpreadTeam.abbr,
      isHome: activeSpreadTeam.isHome, eventId: activeSpreadTeam.eventId,
      spread, gameLabel: game ? _gameLabel(game) : activeSpreadTeam.eventId });

  } else if (activePickType === "total") {
    const eventId = document.getElementById("total-game-select").value;
    const line    = parseFloat(document.getElementById("total-line-input").value);
    if (!eventId || isNaN(line)) { document.getElementById("total-line-input").focus(); return; }
    const game = games.find(g => g.event_id === eventId);
    group.legs.push({ id: Date.now(), type: "total", eventId, line, direction: activeTotalDir,
      gameLabel: game ? _gameLabel(game) : eventId });
  }

  saveParlays();
  closeAddPick();
  renderParlays();

  games.forEach(g => {
    if (!boxscores[g.event_id]) fetchBoxscore(g.event_id).then(renderParlays);
  });
}

// ── Box Score Sidebar ─────────────────────────────────────────────────────
let activeBoxscoreGame = null;

function renderBoxscoreSidebar() {
  const sideEl    = document.getElementById("boxscore-sidebar");
  const labelEl   = document.getElementById("court-game-label");

  // Only show games that actually have player data
  const available = games.filter(g => {
    const bs = boxscores[g.event_id];
    return bs && !bs.error && Object.keys(bs.players || {}).length > 0;
  });
  if (!available.length) {
    const msg = games.length
      ? "No active games right now."
      : "No games today.";
    sideEl.innerHTML = `<div class="bs-empty">${msg}<br><span class="bs-empty-hint">Refreshes every 30s</span></div>`;
    labelEl.textContent = "—";
    return;
  }

  if (!activeBoxscoreGame || !available.find(g => g.event_id === activeBoxscoreGame)) {
    const live = available.find(g => g.status === "STATUS_IN_PROGRESS");
    activeBoxscoreGame = (live || available[0]).event_id;
  }

  // Update nav label
  const currentGame = available.find(g => g.event_id === activeBoxscoreGame);
  if (currentGame) {
    labelEl.textContent = `${currentGame.away_team} @ ${currentGame.home_team}`;
  }

  // Dim nav buttons at edges
  const idx = available.findIndex(g => g.event_id === activeBoxscoreGame);
  document.getElementById("court-prev").style.opacity = idx <= 0 ? "0.3" : "1";
  document.getElementById("court-next").style.opacity = idx >= available.length - 1 ? "0.3" : "1";

  const game = available.find(g => g.event_id === activeBoxscoreGame);
  const bs   = boxscores[activeBoxscoreGame];
  if (!game || !bs) { sideEl.innerHTML = ""; return; }

  const isLive  = bs.is_live;
  const isFinal = bs.is_final;
  const awayScore = bs.scores[game.away_team] ?? "—";
  const homeScore = bs.scores[game.home_team] ?? "—";

  let clockHtml = "";
  if (isLive)       clockHtml = `<div class="court-clock">Q${bs.period} · ${bs.clock}</div>`;
  else if (isFinal) clockHtml = `<div class="court-clock final">FINAL</div>`;

  const pickNames = new Set(parlays.flatMap(g => g.legs).filter(p => p.player).map(p => normName(p.player)));

  // Build team → players, sort by minutes, take top 8 (starters + short bench)
  const byTeam = {};
  for (const [name, p] of Object.entries(bs.players || {})) {
    if (!p.active) continue;
    if (!byTeam[p.team]) byTeam[p.team] = [];
    byTeam[p.team].push({ name, ...p });
  }
  for (const arr of Object.values(byTeam)) {
    arr.sort((a, b) => b.min - a.min);
  }

  const teamOrder = [game.away_team, game.home_team];

  const teamsHtml = teamOrder.map((abbr, ti) => {
    // Try matching by abbr or fallback to first/second key
    const key = Object.keys(byTeam).find(k => k === abbr)
             || Object.keys(byTeam).find(k => k.toLowerCase() === abbr.toLowerCase())
             || Object.keys(byTeam)[ti];
    const players = byTeam[key] || [];
    if (!players.length) return "";

    const logo  = ti === 0 ? game.away_logo : game.home_logo;
    const score = ti === 0 ? awayScore : homeScore;

    const playerChips = players.slice(0, 8).map((p, i) => {
      const isPick    = pickNames.has(normName(p.name));
      const isStarter = p.starter || i < 5;
      const lastName  = p.name.split(" ").slice(1).join(" ") || p.name;
      return `
        <div class="court-player${isPick ? " court-pick" : ""}">
          <div class="court-player-left">
            ${p.headshot
              ? `<img src="${p.headshot}" class="court-headshot" alt="">`
              : `<div class="court-headshot-ph"></div>`}
            <div class="court-player-info">
              <span class="court-name">${lastName}</span>
              <span class="court-mins">${p.min_str || "—"}</span>
            </div>
          </div>
          <div class="court-player-stats">
            <span class="court-pts${p.pts >= 20 ? " hot" : ""}">${p.pts}</span>
            <span class="court-extra">${p.reb}r ${p.ast}a</span>
          </div>
          ${isPick ? `<div class="court-pick-bar"></div>` : ""}
        </div>`;
    }).join("");

    return `
      <div class="court-team">
        <div class="court-team-hdr">
          ${logo ? `<img src="${logo}" class="court-logo" alt="">` : ""}
          <span class="court-abbr">${abbr}</span>
          <span class="court-score">${score}</span>
        </div>
        ${playerChips}
      </div>`;
  }).join(`<div class="court-divider">${clockHtml}</div>`);

  sideEl.innerHTML = teamsHtml || `<div class="bs-empty">Stats not available yet.</div>`;
}

// ── Auto-refresh ──────────────────────────────────────────────────────────
function startRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(async () => {
    await refreshAll();
  }, 30000);
}

// ── Events ────────────────────────────────────────────────────────────────
// Court prev/next
document.getElementById("court-prev").addEventListener("click", () => {
  const available = games.filter(g => { const bs = boxscores[g.event_id]; return bs && !bs.error && Object.keys(bs.players || {}).length > 0; });
  const idx = available.findIndex(g => g.event_id === activeBoxscoreGame);
  if (idx > 0) { activeBoxscoreGame = available[idx - 1].event_id; renderBoxscoreSidebar(); }
});
document.getElementById("court-next").addEventListener("click", () => {
  const available = games.filter(g => { const bs = boxscores[g.event_id]; return bs && !bs.error && Object.keys(bs.players || {}).length > 0; });
  const idx = available.findIndex(g => g.event_id === activeBoxscoreGame);
  if (idx < available.length - 1) { activeBoxscoreGame = available[idx + 1].event_id; renderBoxscoreSidebar(); }
});

document.getElementById("add-pick-btn").addEventListener("click", openAddPick);
document.getElementById("add-group-btn").addEventListener("click", addGroup);
document.getElementById("add-pick-close").addEventListener("click", closeAddPick);
document.getElementById("add-pick-backdrop").addEventListener("click", closeAddPick);
document.getElementById("confirm-pick-btn").addEventListener("click", confirmPick);

document.querySelectorAll(".pick-type-tab").forEach(btn => {
  btn.addEventListener("click", () => _switchPickType(btn.dataset.type));
});

document.querySelectorAll(".pick-stat-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    activeStat = btn.dataset.stat;
    document.querySelectorAll(".pick-stat-tab").forEach(t =>
      t.classList.toggle("active", t === btn));
  });
});

document.querySelectorAll(".pick-dir-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    activeDirection = btn.dataset.dir;
    document.querySelectorAll("#section-prop .pick-dir-tab").forEach(t =>
      t.classList.toggle("active", t === btn));
  });
});

document.querySelectorAll("#total-dir-tabs .pick-dir-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    activeTotalDir = btn.dataset.dir;
    document.querySelectorAll("#total-dir-tabs .pick-dir-tab").forEach(t =>
      t.classList.toggle("active", t === btn));
  });
});

playerInput.addEventListener("input", () => { showSuggestions(playerInput.value); });

document.getElementById("pick-line-input").addEventListener("keydown", e => {
  if (e.key === "Enter") confirmPick();
});

// ── Init ──────────────────────────────────────────────────────────────────
(async () => {
  loadParlays();
  renderParlays();
  // fetchAllHeadshots is fire-and-forget — don't block startRefresh on it
  fetchAllHeadshots();
  await Promise.all([refreshAll(), fetchPlayerList()]);
  startRefresh();
})();
