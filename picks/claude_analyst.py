"""
Claude Analyst — AI-powered betting analysis.

Sends our model predictions, odds, injuries, and patterns to Claude
for final analysis. Claude acts as a senior sports analyst who:
  1. Evaluates each pick from our model
  2. Identifies which picks have the best edge
  3. Flags risks and provides reasoning
  4. Suggests the final slate of bets

Uses the Anthropic Python SDK.

Setup:
  export ANTHROPIC_API_KEY="your_key_here"
"""

import os
import json
from typing import Dict, List, Optional

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    anthropic = None


SYSTEM_PROMPT = """You are an elite NBA sports analyst and betting advisor. You have deep knowledge of:
- NBA team dynamics, coaching strategies, and matchup analysis
- Player roles, usage patterns, and how injuries cascade through a roster
- Statistical models and their limitations
- Bankroll management and value betting principles

You are given:
1. Our ML model's predictions (spread, total, moneyline) for today's games
2. Current betting odds from sportsbooks
3. Injury reports
4. Data-backed performance patterns (how teammates perform when stars are out)
5. Player prop lines and model predictions

Your job:
- Analyze each prediction against the market odds
- Identify where our model sees VALUE (edge over the market)
- Factor in injuries and their ripple effects on player props
- Be HONEST — if our model doesn't have an edge, say so
- Provide clear reasoning for each recommendation

Output format for each pick you recommend:
  PICK: [Team/Player] [Bet Type] [Line] [Over/Under or Side]
  CONFIDENCE: [HIGH/MEDIUM/LOW]
  EDGE: [explanation of why this is a good bet]
  RISK: [what could go wrong]

At the end, provide a FINAL SLATE — your top picks ranked by confidence.
Be selective. Quality over quantity. It's better to pass on a slate than force bad picks."""


def _get_client():
    if anthropic is None:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set.\n"
            "Set it with: export ANTHROPIC_API_KEY='your_key'\n"
            "Get a key at: https://console.anthropic.com/"
        )
    return anthropic.Anthropic(api_key=key)


def analyze_picks(
    model_predictions: str,
    injury_report: str,
    active_patterns: str,
    player_props_summary: str = "",
    odds_summary: str = "",
    target_picks: int = 5,
    previous_analysis: Optional[str] = None,
    iteration: int = 1,
) -> str:
    """
    Send all data to Claude for analysis and get back recommended picks.

    Args:
        model_predictions: Formatted string of model predictions
        injury_report: Formatted injury report
        active_patterns: Formatted injury-driven patterns
        player_props_summary: Formatted player prop data
        odds_summary: Additional odds context
        target_picks: How many picks we want Claude to select
        previous_analysis: Claude's previous response (for iteration)
        iteration: Current iteration number

    Returns:
        Claude's analysis as a string.
    """
    client = _get_client()

    # Build the user message
    parts = []

    if iteration == 1:
        parts.append(f"Analyze today's NBA games and select your TOP {target_picks} picks.\n")
    else:
        parts.append(
            f"ITERATION {iteration}: You previously selected fewer than {target_picks} picks. "
            f"Review again and try to find {target_picks} total picks. "
            f"You can lower your confidence threshold slightly, but don't force bad picks.\n"
            f"\nYour previous analysis:\n{previous_analysis}\n"
            f"\n--- Re-analyze with fresh eyes ---\n"
        )

    parts.append("=" * 60)
    parts.append("MODEL PREDICTIONS")
    parts.append("=" * 60)
    parts.append(model_predictions)

    parts.append("\n" + "=" * 60)
    parts.append("INJURY REPORT")
    parts.append("=" * 60)
    parts.append(injury_report)

    if active_patterns:
        parts.append("\n" + "=" * 60)
        parts.append("INJURY-DRIVEN PERFORMANCE PATTERNS")
        parts.append("=" * 60)
        parts.append(active_patterns)

    if player_props_summary:
        parts.append("\n" + "=" * 60)
        parts.append("PLAYER PROP LINES & PREDICTIONS")
        parts.append("=" * 60)
        parts.append(player_props_summary)

    if odds_summary:
        parts.append("\n" + "=" * 60)
        parts.append("ADDITIONAL ODDS DATA")
        parts.append("=" * 60)
        parts.append(odds_summary)

    parts.append(
        f"\n\nSelect exactly {target_picks} picks (or fewer if you truly can't find value). "
        f"Include a mix of game bets (spread/total/ML) and player props if the data supports it. "
        f"End with a FINAL SLATE section listing your picks."
    )

    user_message = "\n".join(parts)

    print(f"  Sending to Claude (iteration {iteration})...")
    print(f"  Prompt size: ~{len(user_message)} chars")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    analysis = response.content[0].text
    print(f"  Claude responded ({len(analysis)} chars)")

    return analysis


def parse_final_slate(analysis: str) -> List[Dict]:
    """
    Extract structured picks from Claude's FINAL SLATE section.

    Returns list of pick dicts that can be used for tracking/display.
    """
    picks = []

    # Find the FINAL SLATE section
    slate_start = analysis.upper().find("FINAL SLATE")
    if slate_start == -1:
        # Try alternative headers
        for header in ["TOP PICKS", "RECOMMENDED PICKS", "SELECTED PICKS", "MY PICKS"]:
            slate_start = analysis.upper().find(header)
            if slate_start != -1:
                break

    if slate_start == -1:
        return picks

    slate_text = analysis[slate_start:]

    # Parse picks from various formats Claude might use
    for line in slate_text.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()

        if not line_stripped or len(line_stripped) < 8:
            continue

        is_pick = False
        clean = line_stripped

        # Explicit PICK: prefix
        if upper.startswith("PICK") or upper.startswith("- PICK"):
            is_pick = True
            for prefix in ["PICK:", "- PICK:", "PICK "]:
                if clean.upper().startswith(prefix.upper()):
                    clean = clean[len(prefix):].strip()
                    break

        # Numbered picks: "1. BOS -5.5", "1) BOS -5.5"
        elif line_stripped[0].isdigit() and any(line_stripped[1:4].startswith(d) for d in [".", ")", ":"]):
            is_pick = True
            for sep in [".", ")", ":"]:
                if sep in line_stripped[:4]:
                    clean = line_stripped.split(sep, 1)[1].strip()
                    break

        # Bullet points: "- BOS -5.5", "* BOS", "• BOS"
        elif line_stripped[0] in "-*\u2022" and len(line_stripped) > 3:
            is_pick = True
            clean = line_stripped[1:].strip()

        if is_pick and len(clean) > 5:
            picks.append({
                "description": clean,
                "raw_line": line_stripped,
            })

    return picks


def iterative_analysis(
    model_predictions: str,
    injury_report: str,
    active_patterns: str,
    player_props_summary: str = "",
    odds_summary: str = "",
    target_picks: int = 5,
    max_iterations: int = 3,
) -> tuple:
    """
    Run Claude analysis in a loop until we reach target picks or max iterations.

    Returns:
        (final_analysis_text, list_of_picks)
    """
    analysis = None
    picks = []

    for i in range(1, max_iterations + 1):
        analysis = analyze_picks(
            model_predictions=model_predictions,
            injury_report=injury_report,
            active_patterns=active_patterns,
            player_props_summary=player_props_summary,
            odds_summary=odds_summary,
            target_picks=target_picks,
            previous_analysis=analysis if i > 1 else None,
            iteration=i,
        )

        picks = parse_final_slate(analysis)
        print(f"  Iteration {i}: Claude selected {len(picks)} picks (target: {target_picks})")

        if len(picks) >= target_picks:
            break

        if i < max_iterations:
            print(f"  Not enough picks, iterating...")

    return analysis, picks
