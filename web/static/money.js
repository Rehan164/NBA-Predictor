/* NBA Predictor — Money Tracker */

const LS_KEY = "nba_money_txns";
let txns      = [];
let modalType = "deposit"; // "deposit" | "withdraw" | "parlay"

// ── Persistence ───────────────────────────────────────────────────────────
function load() {
  try { txns = JSON.parse(localStorage.getItem(LS_KEY)) || []; } catch { txns = []; }
}
function save() { localStorage.setItem(LS_KEY, JSON.stringify(txns)); }

// ── Helpers ───────────────────────────────────────────────────────────────
function fmt(n) {
  const abs = Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return (n < 0 ? "-$" : "$") + abs;
}
function fmtDate(iso) {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// Convert American odds to total payout (stake + profit)
function calcPayout(stake, odds) {
  if (!stake || !odds || isNaN(odds)) return 0;
  if (odds > 0) return stake + stake * (odds / 100);
  return stake + stake * (100 / Math.abs(odds));
}

// ── Live Tracker integration ──────────────────────────────────────────────
function loadLiveParlays() {
  try { return JSON.parse(localStorage.getItem("nba_live_parlays")) || []; } catch { return []; }
}

const STAT_LABEL = { pts: "PTS", reb: "REB", ast: "AST", tpm: "3PM" };

function legDesc(leg) {
  if (leg.type === "ml")     return `${leg.team} ML`;
  if (leg.type === "spread") return `${leg.team} ${leg.spread >= 0 ? "+" : ""}${leg.spread} ATS`;
  if (leg.type === "total")  return `${leg.gameLabel} ${leg.direction === "over" ? "O" : "U"} ${leg.line}`;
  // prop (default)
  const last = (leg.player || "").split(" ").slice(-1)[0];
  const dir  = leg.direction === "under" ? "U" : "O";
  return `${last} ${STAT_LABEL[leg.stat] || leg.stat} ${dir}${leg.line}`;
}

// ── Balance calculation ───────────────────────────────────────────────────
function calcStats() {
  const deposits    = txns.filter(t => t.type === "deposit");
  const withdrawals = txns.filter(t => t.type === "withdraw");
  const parlays     = txns.filter(t => t.type === "parlay");

  const totalIn     = deposits.reduce((s, t) => s + t.amount, 0);
  const totalOut    = withdrawals.reduce((s, t) => s + t.amount, 0);

  const wonPayouts  = parlays.filter(t => t.status === "won").reduce((s, t) => s + t.payout, 0);
  const wonStakes   = parlays.filter(t => t.status === "won").reduce((s, t) => s + t.stake, 0);
  const lostStakes  = parlays.filter(t => t.status === "lost").reduce((s, t) => s + t.stake, 0);
  const pendStakes  = parlays.filter(t => t.status === "pending").reduce((s, t) => s + t.stake, 0);

  // Balance = deposited − withdrawn − all bet stakes placed + won payouts
  const balance = totalIn - totalOut - wonStakes - lostStakes - pendStakes + wonPayouts;

  // P&L = pure betting performance (deposits/withdrawals are not wins/losses)
  const pnl     = (wonPayouts - wonStakes) - lostStakes;
  const wagered = wonStakes + lostStakes;
  const roi     = wagered > 0 ? (pnl / wagered * 100) : 0;

  return { totalIn, totalOut, balance, pnl, roi, pendStakes,
           parlayCount: parlays.length, pendCount: parlays.filter(t => t.status === "pending").length };
}

// ── Render summary ────────────────────────────────────────────────────────
function renderSummary() {
  const s = calcStats();

  document.getElementById("stat-in").textContent        = fmt(s.totalIn);
  document.getElementById("stat-in-count").textContent  = `${txns.filter(t=>t.type==="deposit").length} deposit(s)`;
  document.getElementById("stat-out").textContent       = fmt(s.totalOut);
  document.getElementById("stat-out-count").textContent = `${txns.filter(t=>t.type==="withdraw").length} withdrawal(s)`;

  const pnlEl = document.getElementById("stat-pnl");
  pnlEl.textContent = fmt(s.pnl);
  pnlEl.className   = "money-card-value" + (s.pnl > 0 ? " positive" : s.pnl < 0 ? " negative" : "");
  document.getElementById("stat-roi").textContent = `${s.roi >= 0 ? "+" : ""}${s.roi.toFixed(1)}% ROI`;

  const balEl = document.getElementById("stat-balance");
  balEl.textContent = fmt(s.balance);
  balEl.className   = "money-card-value" + (s.balance > 0 ? " positive" : s.balance < 0 ? " negative" : "");
  document.getElementById("stat-balance-sub").textContent =
    s.pendCount > 0 ? `${s.pendCount} bet(s) pending` : "on account";
}

// ── Render history ────────────────────────────────────────────────────────
function renderHistory() {
  const el = document.getElementById("history-list");

  if (!txns.length) {
    el.innerHTML = `<div class="history-empty">No transactions yet.<br>Add a deposit, withdrawal, or parlay bet.</div>`;
    return;
  }

  const sorted = [...txns].sort((a, b) => new Date(b.date) - new Date(a.date));

  el.innerHTML = sorted.map(t => {
    if (t.type === "parlay") return renderParlayRow(t);

    const isDeposit = t.type === "deposit";
    return `
      <div class="history-row">
        <div class="history-icon ${t.type}">${isDeposit ? "↓" : "↑"}</div>
        <div class="history-info">
          <div class="history-type">${isDeposit ? "Deposit" : "Withdrawal"}</div>
          <div class="history-meta">${fmtDate(t.date)}${t.note ? " · " + t.note : ""}</div>
        </div>
        <span class="history-amount ${t.type}">${isDeposit ? "+" : "−"}${Math.abs(t.amount).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>
        <button class="history-del" onclick="deleteTxn('${t.id}')" title="Delete">✕</button>
      </div>`;
  }).join("");
}

function renderParlayRow(t) {
  const oddsLabel  = t.odds > 0 ? `+${t.odds}` : `${t.odds}`;
  const profit     = t.payout - t.stake;
  const legsHtml   = t.linkedLegs?.length
    ? `<div class="linked-legs">${t.linkedLegs.map(l => `<span class="linked-leg-chip">${l}</span>`).join("")}</div>`
    : "";
  const linkBadge  = t.linkedName
    ? `<span class="linked-badge">🔗 ${t.linkedName}</span>`
    : "";
  const linkBtn    = !t.linkedGroupId
    ? `<button class="btn-link-parlay" onclick="toggleLinkPanel('${t.id}')">🔗 Link</button>`
    : "";
  const linkPanel  = buildLinkPanel(t.id);

  if (t.status === "pending") {
    return `
      <div class="history-row parlay-row pending" id="prow-${t.id}">
        <div class="history-icon parlay">🎰</div>
        <div class="history-info">
          <div class="history-type">Parlay <span class="odds-pill">${oddsLabel}</span>${linkBadge}${linkBtn}</div>
          <div class="history-meta">${fmtDate(t.date)}${t.note ? " · " + t.note : ""} · to win ${fmt(profit)}</div>
          ${legsHtml}${linkPanel}
        </div>
        <div class="parlay-settle">
          <button class="btn-settle won"  onclick="settleBet('${t.id}','won')">Won</button>
          <button class="btn-settle lost" onclick="settleBet('${t.id}','lost')">Lost</button>
        </div>
        <span class="history-amount parlay-stake">−${t.stake.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>
        <button class="history-del" onclick="deleteTxn('${t.id}')" title="Delete">✕</button>
      </div>`;
  }

  if (t.status === "won") {
    return `
      <div class="history-row parlay-row won" id="prow-${t.id}">
        <div class="history-icon parlay-won">✓</div>
        <div class="history-info">
          <div class="history-type">Parlay <span class="odds-pill">${oddsLabel}</span> <span class="settle-badge won">WON</span>${linkBadge}${linkBtn}</div>
          <div class="history-meta">${fmtDate(t.date)}${t.note ? " · " + t.note : ""} · staked ${fmt(t.stake)}</div>
          ${legsHtml}${linkPanel}
        </div>
        <span class="history-amount deposit">+${profit.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>
        <button class="history-del" onclick="deleteTxn('${t.id}')" title="Delete">✕</button>
      </div>`;
  }

  return `
    <div class="history-row parlay-row lost" id="prow-${t.id}">
      <div class="history-icon parlay-lost">✕</div>
      <div class="history-info">
        <div class="history-type">Parlay <span class="odds-pill">${oddsLabel}</span> <span class="settle-badge lost">LOST</span>${linkBadge}${linkBtn}</div>
        <div class="history-meta">${fmtDate(t.date)}${t.note ? " · " + t.note : ""}</div>
        ${legsHtml}${linkPanel}
      </div>
      <span class="history-amount withdraw">−${t.stake.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>
      <button class="history-del" onclick="deleteTxn('${t.id}')" title="Delete">✕</button>
    </div>`;
}

function buildLinkPanel(txnId) {
  const liveGroups = loadLiveParlays().filter(g => g.legs.length > 0);
  if (!liveGroups.length) return "";
  const options = liveGroups.map(g =>
    `<option value="${g.id}">${g.name} (${g.legs.length} leg${g.legs.length !== 1 ? "s" : ""})</option>`
  ).join("");
  return `
    <div class="link-panel" id="link-panel-${txnId}" style="display:none">
      <select class="link-panel-select form-input" id="link-sel-${txnId}">${options}</select>
      <button class="btn-settle won" onclick="confirmLinkParlay('${txnId}')">Link</button>
      <button class="btn-settle lost" onclick="toggleLinkPanel('${txnId}')">Cancel</button>
    </div>`;
}

function toggleLinkPanel(txnId) {
  const panel = document.getElementById(`link-panel-${txnId}`);
  if (!panel) return;
  panel.style.display = panel.style.display === "none" ? "flex" : "none";
}

function confirmLinkParlay(txnId) {
  const sel   = document.getElementById(`link-sel-${txnId}`);
  if (!sel) return;
  const group = loadLiveParlays().find(g => String(g.id) === sel.value);
  if (!group) return;
  const t = txns.find(t => t.id === txnId);
  if (!t) return;
  t.linkedGroupId = group.id;
  t.linkedName    = group.name;
  t.linkedLegs    = group.legs.map(legDesc);
  save(); render();
}

function render() { renderSummary(); renderHistory(); }

// ── CRUD ──────────────────────────────────────────────────────────────────
function addTxn(type, amount, note) {
  txns.push({ id: Date.now().toString(), type, amount, note: note.trim(), date: new Date().toISOString() });
  save(); render();
}

function addParlay(stake, odds, note, linkedGroupId) {
  let linkedLegs = null;
  let linkedName = null;
  if (linkedGroupId) {
    const group = loadLiveParlays().find(g => String(g.id) === String(linkedGroupId));
    if (group) {
      linkedLegs = group.legs.map(legDesc);
      linkedName = group.name;
    }
  }
  txns.push({ id: Date.now().toString(), type: "parlay", stake, odds,
    payout: calcPayout(stake, odds), note: note.trim(),
    status: "pending", date: new Date().toISOString(),
    linkedGroupId: linkedGroupId || null, linkedLegs, linkedName });
  save(); render();
}

function settleBet(id, result) {
  const t = txns.find(t => t.id === id);
  if (t) { t.status = result; save(); render(); }
}

function deleteTxn(id) {
  txns = txns.filter(t => t.id !== id);
  save(); render();
}

function confirmClearAll() {
  if (txns.length && confirm("Delete all transactions? This cannot be undone.")) {
    txns = []; save(); render();
  }
}

// ── Deposit / Withdraw modal ───────────────────────────────────────────────
function openModal(type) {
  modalType = type;
  const isDeposit = type === "deposit";
  document.getElementById("money-modal-title").textContent = isDeposit ? "Add Deposit" : "Add Withdrawal";
  document.getElementById("money-modal-title").className   = "money-modal-title " + type;
  document.getElementById("money-confirm").textContent     = isDeposit ? "Add Deposit" : "Add Withdrawal";
  document.getElementById("money-confirm").className       = "btn-confirm-money " + type;
  document.getElementById("money-amount").value = "";
  document.getElementById("money-note").value   = "";
  document.getElementById("money-backdrop").classList.add("visible");
  document.getElementById("money-modal").classList.add("visible");
  document.getElementById("money-amount").focus();
}

function closeModal() {
  document.getElementById("money-backdrop").classList.remove("visible");
  document.getElementById("money-modal").classList.remove("visible");
}

function confirmModal() {
  const amount = parseFloat(document.getElementById("money-amount").value);
  const note   = document.getElementById("money-note").value;
  if (!amount || amount <= 0) { document.getElementById("money-amount").focus(); return; }
  addTxn(modalType, amount, note);
  closeModal();
}

// ── Parlay modal ──────────────────────────────────────────────────────────
function openParlayModal() {
  document.getElementById("parlay-stake").value = "";
  document.getElementById("parlay-odds").value  = "";
  document.getElementById("parlay-note").value  = "";
  document.getElementById("parlay-payout-preview").textContent = "—";
  document.getElementById("parlay-profit-preview").textContent = "—";

  // Populate live parlay selector
  const liveGroups = loadLiveParlays();
  const sel        = document.getElementById("parlay-link-select");
  sel.innerHTML    = `<option value="">— None —</option>` +
    liveGroups.filter(g => g.legs.length > 0).map(g =>
      `<option value="${g.id}">${g.name} (${g.legs.length} leg${g.legs.length !== 1 ? "s" : ""})</option>`
    ).join("");
  updateLinkedLegsPreview();

  document.getElementById("parlay-backdrop").classList.add("visible");
  document.getElementById("parlay-modal").classList.add("visible");
  document.getElementById("parlay-stake").focus();
}

function updateLinkedLegsPreview() {
  const sel        = document.getElementById("parlay-link-select");
  const preview    = document.getElementById("parlay-legs-preview");
  const liveGroups = loadLiveParlays();
  const group      = liveGroups.find(g => String(g.id) === sel.value);

  if (!group) { preview.innerHTML = ""; return; }

  preview.innerHTML = group.legs.map(leg =>
    `<span class="linked-leg-chip">${legDesc(leg)}</span>`
  ).join("");
}

function closeParlayModal() {
  document.getElementById("parlay-backdrop").classList.remove("visible");
  document.getElementById("parlay-modal").classList.remove("visible");
}

function updateParlayPreview() {
  const stake = parseFloat(document.getElementById("parlay-stake").value);
  const odds  = parseInt(document.getElementById("parlay-odds").value);
  if (!stake || stake <= 0 || !odds || isNaN(odds)) {
    document.getElementById("parlay-payout-preview").textContent = "—";
    document.getElementById("parlay-profit-preview").textContent = "—";
    return;
  }
  const payout = calcPayout(stake, odds);
  const profit = payout - stake;
  document.getElementById("parlay-payout-preview").textContent = fmt(payout);
  document.getElementById("parlay-profit-preview").textContent = `+${fmt(profit)}`;
}

function confirmParlay() {
  const stake         = parseFloat(document.getElementById("parlay-stake").value);
  const odds          = parseInt(document.getElementById("parlay-odds").value);
  const note          = document.getElementById("parlay-note").value;
  const linkedGroupId = document.getElementById("parlay-link-select").value || null;
  if (!stake || stake <= 0) { document.getElementById("parlay-stake").focus(); return; }
  if (!odds || isNaN(odds)) { document.getElementById("parlay-odds").focus(); return; }
  addParlay(stake, odds, note, linkedGroupId);
  closeParlayModal();
}

// ── Events ────────────────────────────────────────────────────────────────
document.getElementById("btn-deposit").addEventListener("click",  () => openModal("deposit"));
document.getElementById("btn-withdraw").addEventListener("click", () => openModal("withdraw"));
document.getElementById("btn-parlay").addEventListener("click",   openParlayModal);

document.getElementById("money-modal-close").addEventListener("click", closeModal);
document.getElementById("money-backdrop").addEventListener("click", closeModal);
document.getElementById("money-confirm").addEventListener("click", confirmModal);
document.getElementById("money-amount").addEventListener("keydown", e => { if (e.key === "Enter") confirmModal(); });
document.getElementById("money-note").addEventListener("keydown",   e => { if (e.key === "Enter") confirmModal(); });

document.getElementById("parlay-modal-close").addEventListener("click",  closeParlayModal);
document.getElementById("parlay-backdrop").addEventListener("click",     closeParlayModal);
document.getElementById("parlay-confirm").addEventListener("click",      confirmParlay);
document.getElementById("parlay-stake").addEventListener("input",        updateParlayPreview);
document.getElementById("parlay-odds").addEventListener("input",         updateParlayPreview);
document.getElementById("parlay-odds").addEventListener("keydown", e =>  { if (e.key === "Enter") confirmParlay(); });
document.getElementById("parlay-link-select").addEventListener("change", updateLinkedLegsPreview);

// ── Init ──────────────────────────────────────────────────────────────────
load();
render();
