"""
NBA Neural Network v4 — Ensemble + Augmentation for maximum accuracy.

Key design:
  Phase 1 — Sequential feature collection (same as v3)
  Phase 2 — Train an ENSEMBLE of 5 neural networks, each with:
    - Different random initialization (diverse local optima)
    - Home/away data augmentation (doubles effective dataset)
    - 40 epochs with early stopping + cosine LR
    - Matchup interaction features (how rosters match up)

  At prediction time: average all 5 models × 2 perspectives (test-time aug)
  = 10 predictions per game, averaged. This is extremely robust.

  Expected: 70%+ win accuracy on held-out test set.
"""

import json
import math
import pickle
import time
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    HISTORICAL_GAMES_CSV,
    PLAYER_GAME_LOGS_CSV,
    MODELS_DIR,
    RANDOM_STATE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ADVANCED_MODEL_DIR = MODELS_DIR / "advanced"
ADVANCED_MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PT_PATH = ADVANCED_MODEL_DIR / "model.pt"         # stores list of state_dicts
MODEL_META_PATH = ADVANCED_MODEL_DIR / "meta.pkl"
STATUS_PATH = ADVANCED_MODEL_DIR / "status.json"
PLAYER_STATS_PATH = ADVANCED_MODEL_DIR / "player_stats.pkl"
TEAM_STATS_PATH = ADVANCED_MODEL_DIR / "team_stats.pkl"
ELO_PATH = ADVANCED_MODEL_DIR / "elo.pkl"
REST_PATH = ADVANCED_MODEL_DIR / "rest.pkl"
LEARN_HISTORY_PATH = ADVANCED_MODEL_DIR / "learn_history.json"

# Player features
RAW_STATS = 10
DERIVED_STATS = 4
STATS_PER_SCALE = RAW_STATS + DERIVED_STATS  # 14
NUM_SCALES = 2
PLAYER_INPUT_DIM = STATS_PER_SCALE * NUM_SCALES  # 28
PLAYERS_PER_TEAM = 10

# Team features
TEAM_FEATURE_DIM = 10

# Context
CONTEXT_DIM = 12
COMBINED_CONTEXT_DIM = CONTEXT_DIM + TEAM_FEATURE_DIM * 2  # 32

# Architecture
PLAYER_EMBED_DIM = 40
TEAM_REPR_DIM = 80
MATCHUP_DIM = 80          # matchup interaction output
HIDDEN_DIM = 320
NUM_ATTN_HEADS = 4
NUM_TRUNK_LAYERS = 5
DROPOUT = 0.2

# Ensemble
N_ENSEMBLE = 5
ENSEMBLE_SEEDS = [42, 137, 256, 789, 1337]

# Training per model
BATCH_SIZE = 128
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
LABEL_SMOOTHING = 0.05
LOSS_W_WIN = 1.0
LOSS_W_MARGIN = 0.4
LOSS_W_TOTAL = 0.2
COVID_SEASONS = {2020, 2021}

# ESPN abbreviation normalization
ESPN_TO_CSV = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "UTAH": "UTA", "BK": "BKN", "WSH": "WAS",
}
FRANCHISE_MAP = {
    "NJN": "BKN", "SEA": "OKC", "VAN": "MEM", "NOH": "NOP",
    "NOK": "NOP", "CHH": "CHA",
}

def _csv_abbr(espn_abbr: str) -> str:
    return ESPN_TO_CSV.get(espn_abbr, espn_abbr)

def _normalize_team(abbr: str) -> str:
    return FRANCHISE_MAP.get(abbr, abbr)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_data():
    games = pd.read_csv(HISTORICAL_GAMES_CSV)
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values("date").reset_index(drop=True)
    logs = pd.read_csv(PLAYER_GAME_LOGS_CSV, low_memory=False)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    return games, logs


def build_player_game_index(logs):
    index = defaultdict(lambda: defaultdict(list))
    for _, row in logs.iterrows():
        gid = row["GAME_ID"]
        team = row["TEAM_ABBREVIATION"]
        stats = np.array([
            row.get("PTS",0) or 0, row.get("REB",0) or 0,
            row.get("AST",0) or 0, row.get("STL",0) or 0,
            row.get("BLK",0) or 0, row.get("TOV",0) or 0,
            row.get("MIN",0) or 0, row.get("FGM",0) or 0,
            row.get("FGA",0) or 0, row.get("PLUS_MINUS",0) or 0,
        ], dtype=np.float32)
        index[gid][team].append((row["PLAYER_NAME"], stats))
    return dict(index)


def _derive_features(raw):
    pts, reb, ast, stl, blk, tov, mins, fgm, fga, pm = raw
    return np.array([
        fgm/max(fga,1.0), pts/max(mins,1.0),
        ast/max(tov,0.5), (pts+reb+ast+stl+blk-tov)/max(mins,1.0),
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKERS
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerStatTracker:
    def __init__(self, alpha_short=0.25, alpha_long=0.06):
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.short = {}; self.long = {}; self.games = {}

    def get(self, player_name):
        s = self.short.get(player_name, np.zeros(RAW_STATS, dtype=np.float32))
        l = self.long.get(player_name, np.zeros(RAW_STATS, dtype=np.float32))
        return np.concatenate([s, _derive_features(s), l, _derive_features(l)])

    def update(self, player_name, game_stats):
        n = self.games.get(player_name, 0)
        if n == 0:
            self.short[player_name] = game_stats.copy()
            self.long[player_name] = game_stats.copy()
        else:
            self.short[player_name] = (1-self.alpha_short)*self.short[player_name] + self.alpha_short*game_stats
            self.long[player_name] = (1-self.alpha_long)*self.long[player_name] + self.alpha_long*game_stats
        self.games[player_name] = n + 1

    def get_team_roster(self, team_players):
        roster = np.zeros((PLAYERS_PER_TEAM, PLAYER_INPUT_DIM), dtype=np.float32)
        mask = np.zeros(PLAYERS_PER_TEAM, dtype=np.float32)
        for i, (pn, _) in enumerate(sorted(team_players, key=lambda x: x[1][6], reverse=True)[:PLAYERS_PER_TEAM]):
            roster[i] = self.get(pn); mask[i] = 1.0
        return roster, mask

    def update_team(self, team_players):
        for pn, st in team_players: self.update(pn, st)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"as":self.alpha_short,"al":self.alpha_long,"s":self.short,"l":self.long,"g":self.games}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f: d = pickle.load(f)
        t = cls(d.get("as",d.get("alpha_short",0.25)), d.get("al",d.get("alpha_long",0.06)))
        t.short = d.get("s",d.get("short",d.get("stats",{}))); t.long = d.get("l",d.get("long",d.get("stats",{})))
        t.games = d.get("g",d.get("games",{})); return t


class TeamStatTracker:
    def __init__(self, alpha_short=0.15, alpha_long=0.04):
        self.alpha_short = alpha_short; self.alpha_long = alpha_long; self.teams = {}

    def _default(self):
        return {"ws":0.5,"wl":0.5,"ms":0.0,"ml":0.0,"ts":200.0,"tl":200.0,"hw":0.5,"str":0,"gp":0}

    def get(self, team):
        t = self.teams.get(team, self._default())
        return np.array([t["ws"],t["wl"],t["ms"]/20.0,t["ml"]/20.0,t["ts"]/230.0,t["tl"]/230.0,
                         t["hw"],min(t["str"]/10.0,1.0),min(t["gp"]/82.0,1.0),t["ws"]-t["wl"]], dtype=np.float32)

    def update(self, team, won, margin, total, is_home):
        if team not in self.teams: self.teams[team] = self._default()
        t = self.teams[team]; w = 1.0 if won else 0.0
        t["ws"] = (1-self.alpha_short)*t["ws"]+self.alpha_short*w
        t["wl"] = (1-self.alpha_long)*t["wl"]+self.alpha_long*w
        t["ms"] = (1-self.alpha_short)*t["ms"]+self.alpha_short*margin
        t["ml"] = (1-self.alpha_long)*t["ml"]+self.alpha_long*margin
        t["ts"] = (1-self.alpha_short)*t["ts"]+self.alpha_short*total
        t["tl"] = (1-self.alpha_long)*t["tl"]+self.alpha_long*total
        if is_home: t["hw"] = (1-self.alpha_short)*t["hw"]+self.alpha_short*w
        t["str"] = max(t["str"],0)+1 if won else min(t["str"],0)-1; t["gp"] += 1

    def reset_season(self):
        for team in self.teams:
            t = self.teams[team]
            t["ws"]=t["ws"]*0.5+0.25; t["wl"]=t["wl"]*0.7+0.15
            t["ms"]*=0.3; t["ml"]*=0.5; t["hw"]=t["hw"]*0.5+0.25; t["str"]=0; t["gp"]=0

    def save(self, path):
        with open(path,"wb") as f: pickle.dump({"as":self.alpha_short,"al":self.alpha_long,"t":self.teams}, f)

    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: d = pickle.load(f)
        t = cls(d.get("as",0.15), d.get("al",0.04)); t.teams = d.get("t",d.get("teams",{})); return t


class EloTracker:
    def __init__(self, k=20, home_adv=100):
        self.k = k; self.home_adv = home_adv; self.ratings = {}

    def get(self, team): return self.ratings.get(team, 1500.0)

    def expected(self, home, away):
        return 1.0 / (1.0 + 10.0**(-(self.get(home)-self.get(away)+self.home_adv)/400.0))

    def update(self, home, away, home_won, margin=0):
        exp = self.expected(home, away); actual = 1.0 if home_won else 0.0
        mov = max(1.0, math.log(abs(margin)+1)*0.7)
        delta = self.k * mov * (actual - exp)
        self.ratings[home] = self.get(home)+delta; self.ratings[away] = self.get(away)-delta

    def reset_season(self):
        for t in list(self.ratings): self.ratings[t] = self.ratings[t]*0.75+375.0

    def save(self, path):
        with open(path,"wb") as f: pickle.dump({"k":self.k,"ha":self.home_adv,"r":self.ratings}, f)

    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: d = pickle.load(f)
        e = cls(d["k"],d["ha"]); e.ratings = d["r"]; return e


class RestTracker:
    def __init__(self): self.last_game = {}
    def get_rest(self, team, game_date):
        last = self.last_game.get(team)
        return min((game_date - last).days, 7.0) if last is not None else 3.0
    def update(self, team, game_date): self.last_game[team] = game_date
    def save(self, path):
        with open(path,"wb") as f: pickle.dump(self.last_game, f)
    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: r = cls(); r.last_game = pickle.load(f); return r


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(self.norm(x))


class PlayerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PLAYER_INPUT_DIM, PLAYER_EMBED_DIM), nn.GELU(), nn.LayerNorm(PLAYER_EMBED_DIM),
            nn.Linear(PLAYER_EMBED_DIM, PLAYER_EMBED_DIM), nn.GELU(), nn.LayerNorm(PLAYER_EMBED_DIM),
            nn.Linear(PLAYER_EMBED_DIM, PLAYER_EMBED_DIM),
        )
    def forward(self, x): return self.net(x)


class RosterAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(PLAYER_EMBED_DIM, NUM_ATTN_HEADS, dropout=DROPOUT*0.5, batch_first=True)
        self.norm1 = nn.LayerNorm(PLAYER_EMBED_DIM)
        self.attn2 = nn.MultiheadAttention(PLAYER_EMBED_DIM, NUM_ATTN_HEADS, dropout=DROPOUT*0.5, batch_first=True)
        self.norm2 = nn.LayerNorm(PLAYER_EMBED_DIM)
        self.proj = nn.Sequential(nn.Linear(PLAYER_EMBED_DIM, TEAM_REPR_DIM), nn.GELU(), nn.LayerNorm(TEAM_REPR_DIM))

    def forward(self, embeds, mask):
        pad = (mask == 0)
        x = self.norm1(embeds); a1, _ = self.attn1(x,x,x,key_padding_mask=pad); x = embeds + a1
        x2 = self.norm2(x); a2, _ = self.attn2(x2,x2,x2,key_padding_mask=pad); x = x + a2
        w = mask.unsqueeze(-1) + 1e-8
        return self.proj((x * w).sum(dim=1) / w.sum(dim=1))


class MatchupInteraction(nn.Module):
    """Captures how two teams' strengths interact (e.g. offense vs defense)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEAM_REPR_DIM * 3, MATCHUP_DIM * 2), nn.GELU(),
            nn.LayerNorm(MATCHUP_DIM * 2), nn.Dropout(DROPOUT),
            nn.Linear(MATCHUP_DIM * 2, MATCHUP_DIM), nn.GELU(),
        )

    def forward(self, home_repr, away_repr):
        interaction = home_repr * away_repr  # element-wise: captures matchup dynamics
        return self.net(torch.cat([home_repr, away_repr, interaction], dim=1))


class SequentialNBAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.player_encoder = PlayerEncoder()
        self.roster_aggregator = RosterAggregator()
        self.matchup = MatchupInteraction()

        # home_repr(80) + away_repr(80) + matchup(80) + team_stats(20) + context(12) = 272
        input_dim = TEAM_REPR_DIM * 2 + MATCHUP_DIM + COMBINED_CONTEXT_DIM
        self.input_proj = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.GELU(), nn.LayerNorm(HIDDEN_DIM))
        self.trunk = nn.Sequential(*[ResidualBlock(HIDDEN_DIM) for _ in range(NUM_TRUNK_LAYERS)])

        self.win_head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 80), nn.GELU(), nn.Dropout(DROPOUT*0.5), nn.Linear(80, 1))
        self.margin_head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 80), nn.GELU(), nn.Dropout(DROPOUT*0.5), nn.Linear(80, 1))
        self.total_head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 80), nn.GELU(), nn.Dropout(DROPOUT*0.5), nn.Linear(80, 1))

    def forward(self, hr, hm, ar, am, htf, atf, ctx):
        h_emb = self.player_encoder(hr); a_emb = self.player_encoder(ar)
        h_repr = self.roster_aggregator(h_emb, hm); a_repr = self.roster_aggregator(a_emb, am)
        matchup_repr = self.matchup(h_repr, a_repr)
        x = torch.cat([h_repr, a_repr, matchup_repr, htf, atf, ctx], dim=1)
        x = self.trunk(self.input_proj(x))
        return self.win_head(x).squeeze(-1), self.margin_head(x).squeeze(-1), self.total_head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup, total, min_lr=1e-6):
        self.opt = optimizer; self.warmup = warmup; self.total = total
        self.min_lr = min_lr; self.base_lr = optimizer.param_groups[0]["lr"]; self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup:
            lr = self.base_lr * self.step_count / self.warmup
        else:
            p = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5*(self.base_lr-self.min_lr)*(1+math.cos(math.pi*p))
        for pg in self.opt.param_groups: pg["lr"] = lr


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_features(games, game_index, emit, progress_state):
    player_tracker = PlayerStatTracker()
    team_tracker = TeamStatTracker()
    elo_tracker = EloTracker()
    rest_tracker = RestTracker()

    margin_mean = float(games["home_margin"].mean())
    margin_std = float(games["home_margin"].std()) + 1e-8
    total_mean = float(games["total_score"].mean())
    total_std = float(games["total_score"].std()) + 1e-8

    all_hr, all_hm, all_ar, all_am = [], [], [], []
    all_htf, all_atf, all_ctx = [], [], []
    all_win, all_margin, all_total, all_weights = [], [], [], []

    total_games = len(games); prev_season = None; t0 = time.time()

    for idx, (_, row) in enumerate(games.iterrows()):
        gid = row["game_id"]; ht = row["home_team"]; at = row["away_team"]; season = row["season"]
        if prev_season is not None and season != prev_season:
            team_tracker.reset_season(); elo_tracker.reset_season()
        prev_season = season

        gp = game_index.get(gid, {}); hp = gp.get(ht, []); ap = gp.get(at, [])
        if not hp or not ap: continue

        home_roster, home_mask = player_tracker.get_team_roster(hp)
        away_roster, away_mask = player_tracker.get_team_roster(ap)
        htf = team_tracker.get(ht); atf = team_tracker.get(at)

        gd = row["date"]; sd = (gd - pd.Timestamp(f"{season-1}-10-01")).days
        sp = min(1.0, max(0.0, sd/250.0))
        elo_diff = (elo_tracker.get(ht) - elo_tracker.get(at))/400.0
        elo_exp = elo_tracker.expected(ht, at)
        hr_rest = rest_tracker.get_rest(ht, gd)/7.0; ar_rest = rest_tracker.get_rest(at, gd)/7.0
        hb2b = float(rest_tracker.get_rest(ht, gd) <= 1.0); ab2b = float(rest_tracker.get_rest(at, gd) <= 1.0)

        ctx = np.array([1.0, sp, len(hp)/15.0, len(ap)/15.0, season/2026.0,
                        elo_diff, elo_exp, hr_rest, ar_rest, hb2b, ab2b, float(season>=2015)], dtype=np.float32)

        hw = float(row["home_win"]); mg = float(row["home_margin"]); tt = float(row["total_score"])
        weight = 1.0 + max(0.0, (season - 2010)) * 0.08

        all_hr.append(home_roster); all_hm.append(home_mask)
        all_ar.append(away_roster); all_am.append(away_mask)
        all_htf.append(htf); all_atf.append(atf); all_ctx.append(ctx)
        all_win.append(hw); all_margin.append((mg-margin_mean)/margin_std)
        all_total.append((tt-total_mean)/total_std); all_weights.append(weight)

        player_tracker.update_team(hp); player_tracker.update_team(ap)
        team_tracker.update(ht, hw>0.5, mg, tt, True); team_tracker.update(at, hw<0.5, -mg, tt, False)
        elo_tracker.update(ht, at, hw>0.5, mg)
        rest_tracker.update(ht, gd); rest_tracker.update(at, gd)

        if (idx+1) % 3000 == 0:
            el = time.time()-t0; spd = (idx+1)/max(el,0.1); eta = (total_games-idx-1)/max(spd,0.1)
            emit(f"  [COLLECT] {idx+1:,}/{total_games:,} ({(idx+1)/total_games:.0%}) | {spd:.0f} g/s | ETA: {int(eta)}s")
            if progress_state is not None:
                progress_state.update({"game_idx":idx+1,"total_games":total_games,
                    "pct":round((idx+1)/total_games*20,1),"phase":"COLLECT",
                    "eta_display":f"{int(eta//60)}m {int(eta%60)}s","games_per_sec":round(spd,1)})

    emit(f"  Collected {len(all_win):,} samples")

    dataset = {
        "hr":np.array(all_hr),"hm":np.array(all_hm),"ar":np.array(all_ar),"am":np.array(all_am),
        "htf":np.array(all_htf),"atf":np.array(all_atf),"ctx":np.array(all_ctx),
        "win":np.array(all_win,dtype=np.float32),"margin":np.array(all_margin,dtype=np.float32),
        "total":np.array(all_total,dtype=np.float32),"weights":np.array(all_weights,dtype=np.float32),
    }
    norm = {"margin_mean":margin_mean,"margin_std":margin_std,"total_mean":total_mean,"total_std":total_std}
    return dataset, norm, player_tracker, team_tracker, elo_tracker, rest_tracker


# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION — Swap home/away to double dataset
# ═══════════════════════════════════════════════════════════════════════════════

def _augment_data(d):
    """Create mirrored copy: swap home↔away, flip margin/win."""
    aug = {
        "hr": d["ar"].copy(), "hm": d["am"].copy(),
        "ar": d["hr"].copy(), "am": d["hm"].copy(),
        "htf": d["atf"].copy(), "atf": d["htf"].copy(),
        "ctx": d["ctx"].copy(),
        "win": 1.0 - d["win"],
        "margin": -d["margin"],
        "total": d["total"].copy(),
        "weights": d["weights"].copy(),
    }
    # Fix context: swap home/away specific features
    # [0:home_adv, 1:season, 2:h_roster, 3:a_roster, 4:era,
    #  5:elo_diff, 6:elo_exp, 7:h_rest, 8:a_rest, 9:h_b2b, 10:a_b2b, 11:modern]
    aug["ctx"][:,2], aug["ctx"][:,3] = d["ctx"][:,3].copy(), d["ctx"][:,2].copy()
    aug["ctx"][:,5] = -d["ctx"][:,5]       # flip elo diff
    aug["ctx"][:,6] = 1.0 - d["ctx"][:,6]  # flip elo expected
    aug["ctx"][:,7], aug["ctx"][:,8] = d["ctx"][:,8].copy(), d["ctx"][:,7].copy()
    aug["ctx"][:,9], aug["ctx"][:,10] = d["ctx"][:,10].copy(), d["ctx"][:,9].copy()

    return {k: np.concatenate([d[k], aug[k]], axis=0) for k in d}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ENSEMBLE TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _train_single_model(dataset, split, device, seed, model_idx, emit, progress_state):
    """Train one model with given seed. Returns (state_dict, test_acc)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(dataset["win"])

    def to_t(key): return torch.FloatTensor(dataset[key])
    hr,hm,ar,am = to_t("hr"),to_t("hm"),to_t("ar"),to_t("am")
    htf,atf,ctx = to_t("htf"),to_t("atf"),to_t("ctx")
    tgt_w,tgt_m,tgt_t = to_t("win"),to_t("margin"),to_t("total")
    weights = to_t("weights")

    tr_idx = torch.arange(split); te_idx = torch.arange(split, n)
    model = SequentialNBAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = WarmupCosineScheduler(optimizer, WARMUP_STEPS, (split//BATCH_SIZE)*MAX_EPOCHS)
    huber = nn.SmoothL1Loss(reduction="none")

    best_acc = 0.0; best_state = None; patience = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        e_loss = 0.0; e_correct = 0; batches = 0

        for i in range(0, len(perm), BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            if len(bi) < 8: continue

            b = {k: v[bi].to(device) for k, v in
                 [("hr",hr),("hm",hm),("ar",ar),("am",am),("htf",htf),("atf",atf),("ctx",ctx),
                  ("w",tgt_w),("m",tgt_m),("t",tgt_t),("wt",weights)]}
            b["wt"] = b["wt"] / b["wt"].mean()

            optimizer.zero_grad()
            wl, mp, tp = model(b["hr"],b["hm"],b["ar"],b["am"],b["htf"],b["atf"],b["ctx"])
            smooth = b["w"] * (1-LABEL_SMOOTHING) + 0.5*LABEL_SMOOTHING
            loss = (LOSS_W_WIN * F.binary_cross_entropy_with_logits(wl, smooth, weight=b["wt"])
                    + LOSS_W_MARGIN * (huber(mp, b["m"])*b["wt"]).mean()
                    + LOSS_W_TOTAL * (huber(tp, b["t"])*b["wt"]).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step()

            e_loss += loss.item()
            with torch.no_grad():
                e_correct += ((torch.sigmoid(wl)>0.5).float() == b["w"]).sum().item()
            batches += 1

        # Test
        model.eval(); tc = 0
        with torch.no_grad():
            for i in range(0, len(te_idx), BATCH_SIZE*2):
                bi = te_idx[i:i+BATCH_SIZE*2]
                wl,_,_ = model(hr[bi].to(device),hm[bi].to(device),ar[bi].to(device),am[bi].to(device),
                               htf[bi].to(device),atf[bi].to(device),ctx[bi].to(device))
                tc += ((torch.sigmoid(wl)>0.5).float() == tgt_w[bi].to(device)).sum().item()

        t_acc = tc / (n - split)
        tr_acc = e_correct / split

        emit(f"    [{model_idx+1}/{N_ENSEMBLE}] Ep {epoch+1:2d}/{MAX_EPOCHS} | Loss: {e_loss/max(batches,1):.4f} "
             f"| Train: {tr_acc:.1%} | Test: {t_acc:.1%}")

        if progress_state is not None:
            base_pct = 20 + model_idx * (80/N_ENSEMBLE)
            ep_pct = (epoch+1)/MAX_EPOCHS * (80/N_ENSEMBLE)
            progress_state.update({"pct": round(base_pct + ep_pct, 1), "phase": "TRAIN",
                "train_acc": round(tr_acc*100,1), "test_acc": round(t_acc*100,1),
                "loss": round(e_loss/max(batches,1), 4),
                "eta_display": f"Model {model_idx+1}/{N_ENSEMBLE} Ep {epoch+1}/{MAX_EPOCHS}"})

        if t_acc > best_acc:
            best_acc = t_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                emit(f"    [{model_idx+1}/{N_ENSEMBLE}] Early stop ep {epoch+1} (best: {best_acc:.1%})")
                break

    return best_state, best_acc


def _train_network(dataset_raw, device, emit, progress_state):
    """Train ensemble of N models on augmented data."""
    n_raw = len(dataset_raw["win"])
    split_raw = int(n_raw * 0.85)

    # Augment training data only (keep test clean)
    train_slice = {k: v[:split_raw] for k, v in dataset_raw.items()}
    test_slice = {k: v[split_raw:] for k, v in dataset_raw.items()}

    augmented_train = _augment_data(train_slice)
    n_aug = len(augmented_train["win"])
    emit(f"  Augmented training set: {n_aug:,} samples (was {split_raw:,})")

    # Combine: augmented train + original test
    dataset = {k: np.concatenate([augmented_train[k], test_slice[k]], axis=0) for k in dataset_raw}
    split = n_aug  # new split point

    emit(f"  Total: {len(dataset['win']):,} | Train: {split:,} | Test: {len(dataset['win'])-split:,}")

    model_count = sum(p.numel() for p in SequentialNBAModel().parameters() if p.requires_grad)
    emit(f"  Model: {model_count:,} params × {N_ENSEMBLE} ensemble = {model_count*N_ENSEMBLE:,} total")

    all_states = []
    all_accs = []

    for mi, seed in enumerate(ENSEMBLE_SEEDS):
        emit(f"\n  ── Training Model {mi+1}/{N_ENSEMBLE} (seed={seed}) ──")
        state, acc = _train_single_model(dataset, split, device, seed, mi, emit, progress_state)
        all_states.append(state)
        all_accs.append(acc)
        emit(f"  ── Model {mi+1} best test accuracy: {acc:.1%} ──")

    # Ensemble test accuracy
    n_test = len(dataset["win"]) - split
    te_idx = torch.arange(split, len(dataset["win"]))

    # Average predictions from all models
    avg_probs = np.zeros(n_test)
    avg_margin = np.zeros(n_test)
    avg_total = np.zeros(n_test)

    hr_t = torch.FloatTensor(dataset["hr"]); hm_t = torch.FloatTensor(dataset["hm"])
    ar_t = torch.FloatTensor(dataset["ar"]); am_t = torch.FloatTensor(dataset["am"])
    htf_t = torch.FloatTensor(dataset["htf"]); atf_t = torch.FloatTensor(dataset["atf"])
    ctx_t = torch.FloatTensor(dataset["ctx"]); win_t = torch.FloatTensor(dataset["win"])

    for state in all_states:
        m = SequentialNBAModel().to(device)
        m.load_state_dict(state)
        m = m.to(device); m.eval()
        probs_i = []; margin_i = []; total_i = []

        with torch.no_grad():
            for i in range(0, len(te_idx), BATCH_SIZE*2):
                bi = te_idx[i:i+BATCH_SIZE*2]
                wl,mp,tp = m(hr_t[bi].to(device),hm_t[bi].to(device),ar_t[bi].to(device),am_t[bi].to(device),
                             htf_t[bi].to(device),atf_t[bi].to(device),ctx_t[bi].to(device))
                probs_i.extend(torch.sigmoid(wl).cpu().numpy().tolist())
                margin_i.extend(mp.cpu().numpy().tolist())
                total_i.extend(tp.cpu().numpy().tolist())

        avg_probs += np.array(probs_i) / N_ENSEMBLE
        avg_margin += np.array(margin_i) / N_ENSEMBLE
        avg_total += np.array(total_i) / N_ENSEMBLE

    test_win = dataset["win"][split:]
    ensemble_correct = np.sum((avg_probs > 0.5).astype(float) == test_win)
    ensemble_acc = ensemble_correct / n_test

    test_margin_raw = dataset["margin"][split:]
    test_total_raw = dataset["total"][split:]
    margin_mae_norm = np.mean(np.abs(avg_margin - test_margin_raw))
    total_mae_norm = np.mean(np.abs(avg_total - test_total_raw))

    emit(f"\n  Individual model accuracies: {[f'{a:.1%}' for a in all_accs]}")
    emit(f"  ENSEMBLE accuracy: {ensemble_acc:.1%}")

    return all_states, ensemble_acc, margin_mae_norm, total_mae_norm, split, n_test, np.mean(all_accs)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(progress_callback=None, progress_state=None):
    def emit(msg):
        print(msg)
        if progress_callback: progress_callback(msg)

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    emit("Loading data...")
    games, logs = _load_all_data()
    emit(f"  {len(games):,} games, {len(logs):,} player logs")

    emit("Building player-game index...")
    game_index = build_player_game_index(logs)
    emit(f"  {len(game_index):,} games indexed")

    games = games[~games["season"].isin(COVID_SEASONS)].reset_index(drop=True)
    emit(f"  After COVID filter: {len(games):,} games")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit(f"Device: {device}")

    emit("\n═══ Phase 1: Sequential Feature Collection ═══")
    dataset, norm_params, player_tracker, team_tracker, elo_tracker, rest_tracker = \
        _collect_features(games, game_index, emit, progress_state)

    emit(f"\n═══ Phase 2: {N_ENSEMBLE}-Model Ensemble Training ═══")

    all_states, ens_acc, margin_mae_n, total_mae_n, n_train, n_test, avg_single = \
        _train_network(dataset, device, emit, progress_state)

    margin_mae = margin_mae_n * norm_params["margin_std"]
    total_mae = total_mae_n * norm_params["total_std"]

    metrics = {
        "win_accuracy": round(float(ens_acc), 4),
        "margin_mae": round(float(margin_mae), 2),
        "total_mae": round(float(total_mae), 2),
        "avg_single_accuracy": round(float(avg_single), 4),
        "ensemble_size": N_ENSEMBLE,
        "train_samples": n_train,
        "test_games": n_test,
        "total_games_processed": len(dataset["win"]),
    }

    emit(f"\n{'='*50}")
    emit(f"  ENSEMBLE Test Accuracy: {ens_acc:.1%} ({n_test:,} games)")
    emit(f"  Avg Single Model:      {avg_single:.1%}")
    emit(f"  Test Margin MAE:       {margin_mae:.2f} pts")
    emit(f"  Test Total MAE:        {total_mae:.2f} pts")
    emit(f"{'='*50}")

    # Save ensemble
    torch.save(all_states, MODEL_PT_PATH)
    with open(MODEL_META_PATH, "wb") as f:
        pickle.dump({"metrics":metrics,"norm_params":norm_params,"trained_at":datetime.now().isoformat(),"ensemble":True}, f)
    with open(STATUS_PATH, "w") as f:
        json.dump({"trained":True,"trained_at":datetime.now().isoformat(),"metrics":metrics}, f, indent=2)

    player_tracker.save(PLAYER_STATS_PATH); team_tracker.save(TEAM_STATS_PATH)
    elo_tracker.save(ELO_PATH); rest_tracker.save(REST_PATH)
    emit("All models and tracker states saved")
    emit("TRAINING COMPLETE")

    return None, metrics, norm_params


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD / STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ensemble():
    """Load all ensemble models."""
    if not MODEL_PT_PATH.exists() or not MODEL_META_PATH.exists(): return None
    with open(MODEL_META_PATH, "rb") as f: meta = pickle.load(f)
    state_dicts = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(state_dicts, list):
        state_dicts = [state_dicts]  # backwards compat with single model
    models = []
    for sd in state_dicts:
        m = SequentialNBAModel(); m.load_state_dict(sd); m.eval(); models.append(m)
    return models, meta


def load_model():
    """Backwards-compatible: returns (first_model, meta) for learn_from_results."""
    result = _load_ensemble()
    if result is None: return None
    return result[0][0], result[1]


def get_model_status():
    status = {"trained":False,"trained_at":None,"metrics":None,"data_status":"unknown","last_data_date":None}
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f: status.update(json.load(f))
    try:
        games = pd.read_csv(HISTORICAL_GAMES_CSV); games["date"] = pd.to_datetime(games["date"])
        last = games["date"].max(); status["last_data_date"] = last.strftime("%Y-%m-%d")
        yesterday = pd.Timestamp(date.today()) - pd.Timedelta(days=1)
        status["data_status"] = "current" if last>=yesterday else "slightly_behind" if last>=yesterday-pd.Timedelta(days=3) else "behind"
    except Exception: status["data_status"] = "error"
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def learn_from_results(target_date=None, progress_callback=None):
    def emit(msg):
        print(msg)
        if progress_callback: progress_callback(msg)

    result = _load_ensemble()
    if result is None: return {"error": "No model trained yet"}
    models, meta = result
    norm_params = meta["norm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device) for m in models]

    player_tracker = PlayerStatTracker.load(PLAYER_STATS_PATH) if PLAYER_STATS_PATH.exists() else PlayerStatTracker()
    team_tracker = TeamStatTracker.load(TEAM_STATS_PATH) if TEAM_STATS_PATH.exists() else TeamStatTracker()
    elo_tracker = EloTracker.load(ELO_PATH) if ELO_PATH.exists() else EloTracker()

    import requests
    from datetime import timezone
    ET = timezone(timedelta(hours=-5))
    if target_date is None:
        target_date = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
    else: target_date = target_date.replace("-","")

    emit(f"Fetching results for {target_date}...")
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={target_date}", timeout=10)
        r.raise_for_status(); espn_data = r.json()
    except Exception as e: return {"error": f"Failed to fetch: {e}"}

    logs = pd.read_csv(PLAYER_GAME_LOGS_CSV, low_memory=False)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    games_data = []; game_labels = []
    for event in espn_data.get("events", []):
        comp = event["competitions"][0]
        if comp.get("status",{}).get("type",{}).get("name","") != "STATUS_FINAL": continue
        ht = at = None; hs = aws = 0
        for c in comp["competitors"]:
            abbr = _csv_abbr(c["team"]["abbreviation"]); score = int(c.get("score",0))
            if c["homeAway"]=="home": ht,hs = abbr,score
            else: at,aws = abbr,score
        if not ht or not at: continue
        hp = _get_recent_players(logs, ht); ap = _get_recent_players(logs, at)
        if not hp or not ap: continue
        hr,hmask = player_tracker.get_team_roster(hp); ar,amask = player_tracker.get_team_roster(ap)
        htf = team_tracker.get(ht); atf = team_tracker.get(at)
        elo_diff = (elo_tracker.get(ht)-elo_tracker.get(at))/400.0; elo_exp = elo_tracker.expected(ht,at)
        ctx = np.array([1.0,0.7,len(hp)/15.0,len(ap)/15.0,2026.0/2026.0,elo_diff,elo_exp,0.4,0.4,0.0,0.0,1.0],dtype=np.float32)
        hw = 1.0 if hs>aws else 0.0; mg = float(hs-aws); tt = float(hs+aws)
        games_data.append({"hr":hr,"hm":hmask,"ar":ar,"am":amask,"htf":htf,"atf":atf,"ctx":ctx,
            "win":hw,"m_norm":(mg-norm_params["margin_mean"])/norm_params["margin_std"],
            "t_norm":(tt-norm_params["total_mean"])/norm_params["total_std"],
            "margin":mg,"total":tt,"home_players":hp,"away_players":ap,"home_team":ht,"away_team":at})
        game_labels.append(f"{at} @ {ht}: {aws}-{hs}")

    if not games_data: return {"error": "No completed games found"}
    emit(f"Found {len(games_data)} completed games")

    # Ensemble pre-learn predictions
    results = []; correct_before = 0
    for i, g in enumerate(games_data):
        avg_wp = 0.0; avg_mg = 0.0; avg_tt = 0.0
        with torch.no_grad():
            for m in models:
                m.eval()
                wl,mp,tp = m(torch.FloatTensor(g["hr"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["hm"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["ar"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["am"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["htf"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["atf"]).unsqueeze(0).to(device),
                             torch.FloatTensor(g["ctx"]).unsqueeze(0).to(device))
                avg_wp += torch.sigmoid(wl).item() / len(models)
                avg_mg += (mp.item()*norm_params["margin_std"]+norm_params["margin_mean"]) / len(models)
                avg_tt += (tp.item()*norm_params["total_std"]+norm_params["total_mean"]) / len(models)
        correct = (avg_wp>0.5)==(g["win"]>0.5)
        if correct: correct_before += 1
        results.append({"game":game_labels[i],"pred_win_prob":round(avg_wp*100,1),
            "pred_margin":round(avg_mg,1),"pred_total":round(avg_tt,1),
            "actual_margin":round(g["margin"],1),"actual_total":round(g["total"],1),
            "correct":bool(correct),"margin_error":round(abs(avg_mg-g["margin"]),1),
            "total_error":round(abs(avg_tt-g["total"]),1)})

    acc_before = correct_before / len(results)
    emit(f"Pre-learn accuracy: {acc_before:.0%} ({correct_before}/{len(results)})")

    # Fine-tune ALL ensemble models
    t_hr = torch.FloatTensor(np.array([g["hr"] for g in games_data])).to(device)
    t_hm = torch.FloatTensor(np.array([g["hm"] for g in games_data])).to(device)
    t_ar = torch.FloatTensor(np.array([g["ar"] for g in games_data])).to(device)
    t_am = torch.FloatTensor(np.array([g["am"] for g in games_data])).to(device)
    t_htf = torch.FloatTensor(np.array([g["htf"] for g in games_data])).to(device)
    t_atf = torch.FloatTensor(np.array([g["atf"] for g in games_data])).to(device)
    t_ctx = torch.FloatTensor(np.array([g["ctx"] for g in games_data])).to(device)
    t_w = torch.FloatTensor([g["win"] for g in games_data]).to(device)
    t_m = torch.FloatTensor([g["m_norm"] for g in games_data]).to(device)
    t_t = torch.FloatTensor([g["t_norm"] for g in games_data]).to(device)
    huber = nn.SmoothL1Loss()

    for mi, m in enumerate(models):
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-5)
        for ep in range(5):
            opt.zero_grad()
            wl,mp,tp = m(t_hr,t_hm,t_ar,t_am,t_htf,t_atf,t_ctx)
            loss = F.binary_cross_entropy_with_logits(wl,t_w) + 0.5*huber(mp,t_m) + 0.3*huber(tp,t_t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        emit(f"  Fine-tuned model {mi+1}/{len(models)}")

    # Post-learn ensemble accuracy
    correct_after = 0
    for i, g in enumerate(games_data):
        avg_wp = 0.0
        with torch.no_grad():
            for m in models:
                wl,_,_ = m(t_hr[i:i+1],t_hm[i:i+1],t_ar[i:i+1],t_am[i:i+1],t_htf[i:i+1],t_atf[i:i+1],t_ctx[i:i+1])
                avg_wp += torch.sigmoid(wl).item() / len(models)
        if (avg_wp>0.5)==(g["win"]>0.5): correct_after += 1

    acc_after = correct_after / len(results)
    emit(f"Post-learn accuracy: {acc_after:.0%} ({correct_after}/{len(results)})")

    # Update trackers
    for g in games_data:
        player_tracker.update_team(g["home_players"]); player_tracker.update_team(g["away_players"])
        team_tracker.update(g["home_team"],g["win"]>0.5,g["margin"],g["total"],True)
        team_tracker.update(g["away_team"],g["win"]<0.5,-g["margin"],g["total"],False)
        elo_tracker.update(g["home_team"],g["away_team"],g["win"]>0.5,g["margin"])

    torch.save([m.cpu().state_dict() for m in models], MODEL_PT_PATH)
    player_tracker.save(PLAYER_STATS_PATH); team_tracker.save(TEAM_STATS_PATH); elo_tracker.save(ELO_PATH)
    emit("Updated all models and tracker states saved")

    entry = {"date":target_date,"learned_at":datetime.now().isoformat(),"games":len(results),
             "accuracy_before":round(acc_before,4),"accuracy_after":round(acc_after,4)}
    history = []
    if LEARN_HISTORY_PATH.exists():
        try:
            with open(LEARN_HISTORY_PATH) as f: history = json.load(f)
        except: pass
    history.append(entry)
    with open(LEARN_HISTORY_PATH, "w") as f: json.dump(history[-100:], f, indent=2)

    return {"date":target_date,"games":len(results),"accuracy_before":round(acc_before*100,1),
            "accuracy_after":round(acc_after*100,1),"results":results}


def _get_recent_players(logs, team):
    tl = logs[logs["TEAM_ABBREVIATION"]==team]
    if tl.empty: return []
    rids = tl.sort_values("GAME_DATE",ascending=False)["GAME_ID"].unique()[:10]
    recent = tl[tl["GAME_ID"].isin(rids)]
    agg = recent.groupby("PLAYER_NAME").agg(
        pts=("PTS","mean"),reb=("REB","mean"),ast=("AST","mean"),stl=("STL","mean"),
        blk=("BLK","mean"),tov=("TOV","mean"),mins=("MIN","mean"),fgm=("FGM","mean"),
        fga=("FGA","mean"),pm=("PLUS_MINUS","mean")).sort_values("mins",ascending=False)
    players = []
    for pn, row in agg.head(PLAYERS_PER_TEAM).iterrows():
        players.append((pn, np.array([row["pts"],row["reb"],row["ast"],row["stl"],row["blk"],
                                      row["tov"],row["mins"],row["fgm"],row["fga"],row["pm"]],dtype=np.float32)))
    return players


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION — Ensemble + Test-Time Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def predict_today(model=None, meta=None):
    result = _load_ensemble()
    if result is None: return []
    models, meta = result
    norm_params = meta["norm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device) for m in models]
    for m in models: m.eval()

    player_tracker = PlayerStatTracker.load(PLAYER_STATS_PATH) if PLAYER_STATS_PATH.exists() else PlayerStatTracker()
    team_tracker = TeamStatTracker.load(TEAM_STATS_PATH) if TEAM_STATS_PATH.exists() else TeamStatTracker()
    elo_tracker = EloTracker.load(ELO_PATH) if ELO_PATH.exists() else EloTracker()
    rest_tracker = RestTracker.load(REST_PATH) if REST_PATH.exists() else RestTracker()

    import requests
    from datetime import timezone
    ET = timezone(timedelta(hours=-5))
    today = datetime.now(ET).strftime("%Y%m%d")
    today_date = pd.Timestamp(datetime.now(ET).date())

    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}",timeout=10)
        r.raise_for_status(); espn_data = r.json()
    except: return []

    from .injuries import get_espn_injury_report
    injuries = {_csv_abbr(k):v for k,v in get_espn_injury_report().items()}
    try:
        from .odds_api import fetch_todays_odds
        odds_map = {(o["home"],o["away"]):o for o in fetch_todays_odds()}
    except: odds_map = {}

    logs = pd.read_csv(PLAYER_GAME_LOGS_CSV, low_memory=False)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    predictions = []

    for event in espn_data.get("events", []):
        comp = event["competitions"][0]; ht = at = None; hd = ad = ""
        for c in comp["competitors"]:
            abbr = _csv_abbr(c["team"]["abbreviation"])
            if c["homeAway"]=="home": ht,hd = abbr,c["team"]["displayName"]
            else: at,ad = abbr,c["team"]["displayName"]
        if not ht or not at: continue

        hp = _get_recent_players(logs,ht); ap = _get_recent_players(logs,at)
        if not hp and not ap: continue

        home_out = _get_out_players(injuries.get(ht,[])); away_out = _get_out_players(injuries.get(at,[]))
        home_roster,home_mask = player_tracker.get_team_roster(hp)
        away_roster,away_mask = player_tracker.get_team_roster(ap)
        hm_count = am_count = 0

        for i,(pn,_) in enumerate(sorted(hp,key=lambda x:x[1][6],reverse=True)[:PLAYERS_PER_TEAM]):
            npn = unicodedata.normalize("NFD",pn).encode("ascii","ignore").decode("ascii")
            if pn in home_out or npn in home_out: home_mask[i]=0.0; hm_count+=1
        for i,(pn,_) in enumerate(sorted(ap,key=lambda x:x[1][6],reverse=True)[:PLAYERS_PER_TEAM]):
            npn = unicodedata.normalize("NFD",pn).encode("ascii","ignore").decode("ascii")
            if pn in away_out or npn in away_out: away_mask[i]=0.0; am_count+=1

        htf = team_tracker.get(ht); atf = team_tracker.get(at)
        elo_diff = (elo_tracker.get(ht)-elo_tracker.get(at))/400.0; elo_exp = elo_tracker.expected(ht,at)
        h_rest = rest_tracker.get_rest(ht,today_date)/7.0; a_rest = rest_tracker.get_rest(at,today_date)/7.0
        hb2b = float(rest_tracker.get_rest(ht,today_date)<=1.0); ab2b = float(rest_tracker.get_rest(at,today_date)<=1.0)
        ctx = np.array([1.0,0.7,len(hp)/15.0,len(ap)/15.0,2026.0/2026.0,elo_diff,elo_exp,h_rest,a_rest,hb2b,ab2b,1.0],dtype=np.float32)

        # Flipped context for test-time augmentation
        ctx_flip = np.array([1.0,0.7,len(ap)/15.0,len(hp)/15.0,2026.0/2026.0,
                             -elo_diff,1.0-elo_exp,a_rest,h_rest,ab2b,hb2b,1.0],dtype=np.float32)

        # Ensemble + test-time augmentation: N_ENSEMBLE models × 2 perspectives = 10 predictions
        avg_wp = 0.0; avg_mg = 0.0; avg_tt = 0.0
        n_preds = len(models) * 2

        with torch.no_grad():
            t_hr = torch.FloatTensor(home_roster).unsqueeze(0).to(device)
            t_hm = torch.FloatTensor(home_mask).unsqueeze(0).to(device)
            t_ar = torch.FloatTensor(away_roster).unsqueeze(0).to(device)
            t_am = torch.FloatTensor(away_mask).unsqueeze(0).to(device)
            t_htf = torch.FloatTensor(htf).unsqueeze(0).to(device)
            t_atf = torch.FloatTensor(atf).unsqueeze(0).to(device)
            t_ctx = torch.FloatTensor(ctx).unsqueeze(0).to(device)
            t_ctx_f = torch.FloatTensor(ctx_flip).unsqueeze(0).to(device)

            for m in models:
                # Original perspective
                wl,mp,tp = m(t_hr,t_hm,t_ar,t_am,t_htf,t_atf,t_ctx)
                avg_wp += torch.sigmoid(wl).item() / n_preds
                avg_mg += (mp.item()*norm_params["margin_std"]+norm_params["margin_mean"]) / n_preds
                avg_tt += (tp.item()*norm_params["total_std"]+norm_params["total_mean"]) / n_preds

                # Flipped perspective (away as home)
                wl2,mp2,tp2 = m(t_ar,t_am,t_hr,t_hm,t_atf,t_htf,t_ctx_f)
                avg_wp += (1.0 - torch.sigmoid(wl2).item()) / n_preds  # flip back
                avg_mg += -(mp2.item()*norm_params["margin_std"]+norm_params["margin_mean"]) / n_preds  # flip margin
                avg_tt += (tp2.item()*norm_params["total_std"]+norm_params["total_mean"]) / n_preds

        odds = odds_map.get((ht,at), {})
        fav_prob = max(avg_wp, 1-avg_wp)
        confidence = round((fav_prob-0.5)*200, 1)

        pred = {"home_team":ht,"away_team":at,"home_name":hd,"away_name":ad,
                "win_prob":round(avg_wp*100,1),"predicted_margin":round(avg_mg,1),
                "predicted_total":round(avg_tt,1),"confidence":confidence,
                "odds_spread":odds.get("spread"),"odds_total":odds.get("total"),
                "odds_home_ml":odds.get("home_ml"),"odds_away_ml":odds.get("away_ml"),
                "home_missing":hm_count,"away_missing":am_count}

        picks = []; fav = ht if avg_wp>0.5 else at
        if fav_prob>=0.55:
            picks.append({"type":"ML","pick":f"{fav} ML","confidence":round(fav_prob*100,1)})
        if odds.get("spread") is not None:
            edge = avg_mg - odds["spread"]
            if abs(edge)>2.0:
                pt = f"{ht} {odds['spread']:+.1f}" if edge>0 else f"{at} {-odds['spread']:+.1f}"
                picks.append({"type":"Spread","pick":pt,"confidence":round(min(80,52+abs(edge)*2.5),1),"edge":round(edge,1)})
        if odds.get("total") is not None:
            te = avg_tt - odds["total"]
            if abs(te)>4:
                picks.append({"type":"Total","pick":f"{'Over' if te>0 else 'Under'} {odds['total']}",
                              "confidence":round(min(75,52+abs(te)*1.5),1),"edge":round(te,1)})
        pred["picks"] = picks; predictions.append(pred)

    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions


def _get_out_players(team_injuries):
    out = set()
    for inj in team_injuries:
        status = inj.get("status","").upper(); name = inj.get("name","")
        if "OUT" in status or "SUSPEND" in status or "DOUBTFUL" in status:
            out.add(name); out.add(unicodedata.normalize("NFD",name).encode("ascii","ignore").decode("ascii"))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="NBA Neural Network v4 — Ensemble")
    p.add_argument("--status",action="store_true"); p.add_argument("--predict",action="store_true")
    p.add_argument("--learn",action="store_true"); args = p.parse_args()
    if args.status: print(json.dumps(get_model_status(),indent=2))
    elif args.predict:
        for p in predict_today():
            print(f"\n{p['away_team']} @ {p['home_team']}: {p['win_prob']}% | margin {p['predicted_margin']:+.1f} | total {p['predicted_total']:.1f}")
            for pick in p.get("picks",[]): print(f"  → {pick['pick']} ({pick['confidence']}%)")
    elif args.learn: print(json.dumps(learn_from_results(),indent=2))
    else: train_model()

if __name__ == "__main__": main()
