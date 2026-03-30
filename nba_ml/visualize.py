"""
Preliminary Data Observations

Two scatter plots with linear regression lines showing
key predictors of Underdog Fantasy points:
    PTS x1  |  AST x1.5  |  REB x1.2  |  STL x3  |  BLK x3  |  TOV x-1

Usage:
    python -m nba_ml.visualize
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .config import DATA_DIR, HISTORICAL_GAMES_CSV, PLAYER_GAME_LOGS_CSV


def calc_fantasy_pts(row):
    return (
        row["PTS"] * 1.0
        + row["AST"] * 1.5
        + row["REB"] * 1.2
        + row["STL"] * 3.0
        + row["BLK"] * 3.0
        + row["TOV"] * -1.0
    )


def main():
    print("  Loading data...")
    plr = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    plr["GAME_DATE"] = pd.to_datetime(plr["GAME_DATE"])
    for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FGA"]:
        plr[col] = pd.to_numeric(plr[col], errors="coerce").fillna(0)
    plr["FPTS"] = calc_fantasy_pts(plr)

    # Only players who actually played
    df = plr[plr["MIN"] > 0].copy()
    sample = df.sample(min(8000, len(df)), random_state=42)

    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Plot 1: Minutes Played vs Fantasy Points ──
    ax1.scatter(sample["MIN"], sample["FPTS"], alpha=0.15, s=10, c="#4fc3f7", edgecolors="none")
    # Linear fit
    m1, b1 = np.polyfit(df["MIN"], df["FPTS"], 1)
    x_line = np.array([0, 48])
    ax1.plot(x_line, m1 * x_line + b1, color="#e94560", linewidth=2.5, label=f"y = {m1:.2f}x + {b1:.2f}")
    r1 = df[["MIN", "FPTS"]].corr().iloc[0, 1]
    ax1.set_xlabel("Minutes Played", fontsize=12)
    ax1.set_ylabel("Fantasy Points (Underdog)", fontsize=12)
    ax1.set_title(f"Minutes Played vs Fantasy Points   (r = {r1:.3f})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)

    # ── Plot 2: Field Goal Attempts vs Fantasy Points ──
    fga_df = df[df["FGA"] > 0].copy()
    fga_sample = fga_df.sample(min(8000, len(fga_df)), random_state=42)
    ax2.scatter(fga_sample["FGA"], fga_sample["FPTS"], alpha=0.15, s=10, c="#4fc3f7", edgecolors="none")
    # Linear fit
    m2, b2 = np.polyfit(fga_df["FGA"], fga_df["FPTS"], 1)
    x_line2 = np.array([0, 35])
    ax2.plot(x_line2, m2 * x_line2 + b2, color="#e94560", linewidth=2.5, label=f"y = {m2:.2f}x + {b2:.2f}")
    r2 = fga_df[["FGA", "FPTS"]].corr().iloc[0, 1]
    ax2.set_xlabel("Field Goal Attempts", fontsize=12)
    ax2.set_ylabel("Fantasy Points (Underdog)", fontsize=12)
    ax2.set_title(f"Field Goal Attempts vs Fantasy Points   (r = {r2:.3f})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    out = DATA_DIR / "scatter_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
