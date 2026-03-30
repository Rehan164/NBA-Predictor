"""
NBA Picks System — Claude-powered betting analysis.

Pipeline:
  1. Fetch today's games, odds (The-Odds-API), and injuries (ESPN)
  2. Run our trained ML models for predictions
  3. Detect injury-driven opportunity patterns
  4. Send favorable plays to Claude for final analysis
  5. Iterate until we reach target number of picks
"""
