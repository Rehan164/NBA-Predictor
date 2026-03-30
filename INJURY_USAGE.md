# Injury Report Integration Guide

## Overview

The prediction system now automatically fetches and applies injury adjustments to spreads, totals, and player props. When star players are OUT, predictions are adjusted based on historical impact.

## How It Works

### Automatic Injury Fetching

By default, the system fetches the latest injury report from ESPN:

```bash
python -m nba_ml.predict
```

The system will:
1. Fetch current injury reports from ESPN API
2. Filter to only OUT players (not questionable/doubtful)
3. Apply adjustments based on player impact values
4. Show adjusted predictions with injury notes

### Manual Injury Input

For today's example (Luka and LeBron out):

```bash
python -m nba_ml.predict --injuries="LAL:LeBron James,Anthony Davis|DAL:Luka Doncic"
```

**Format:** `TEAM:Player1,Player2|TEAM2:Player3`

More examples:
```bash
# Single team, single player
python -m nba_ml.predict --injuries="GSW:Stephen Curry"

# Multiple teams
python -m nba_ml.predict --injuries="MIL:Giannis Antetokounmpo|BOS:Jayson Tatum"

# Multiple players on same team
python -m nba_ml.predict --injuries="PHX:Kevin Durant,Devin Booker"
```

## Injury Impact Values

### Star Players (Configured in `injuries.py`)

| Player | Spread Impact | Total Impact | Usage Shift |
|--------|---------------|--------------|-------------|
| **Luka Doncic** | -5.5 pts | -7.0 pts | +18% to teammates |
| **LeBron James** | -4.5 pts | -6.0 pts | +15% to teammates |
| **Giannis Antetokounmpo** | -6.0 pts | -7.5 pts | +20% to teammates |
| **Nikola Jokic** | -5.5 pts | -7.0 pts | +18% to teammates |
| **Stephen Curry** | -5.0 pts | -6.5 pts | +16% to teammates |

Default starter (non-star): -1.5 spread, -2.5 total

## Example Output

```
  INJURY REPORT (OUT):
  --------------------------------------------------
  DAL:
    • Luka Doncic ⭐
  LAL:
    • LeBron James ⭐

  DAL @ SAS  7:30 PM ET
  --------------------------------------------------------
  Lines:  Spread: SAS -3.5  |  O/U: 220.5  |  ML: SAS -165 / DAL +140
  Injury: Spread adj: +5.5 | Total adj: -7.0
  OUT:    DAL: Luka Doncic
  Model:  Model Margin: +2.0 | Model Total: 213

    Spread    SAS -3.5                     Conf:  72.3% [Strong  ]  (-110) <<
    O/U       UNDER 220.5                  Conf:  68.5% [Strong  ]  (-110) <<
```

## How Injuries Affect Predictions

### Team Predictions

**Spread:**
- Home team loses star → spread moves AGAINST home (harder to cover)
- Away team loses star → spread moves TOWARD home (easier to cover)
- Example: DAL @ SAS, Luka out (-5.5) → SAS spread improves by 5.5 pts

**Total:**
- ANY team losing stars → total goes DOWN
- Both teams with injuries → larger total reduction
- Example: If both LeBron and Luka out → total drops ~13 pts

**Moneyline:**
- Indirectly affected through spread adjustments
- Confidence penalties applied for uncertainty

### Player Props

**Direct Impact - OUT Player:**
- Player's props unavailable (obviously)

**Indirect Impact - Teammates:**
When stars are out, remaining players get usage boost:

**Lakers without LeBron:**
- Anthony Davis: +20% usage → higher PTS/REB props
- Austin Reaves: +25% usage → higher PTS/AST props
- D'Angelo Russell: +20% usage → higher PTS props

**Mavericks without Luka:**
- Kyrie Irving: +25% usage → significantly higher across all props
- PJ Washington: +20% usage → higher PTS/REB props
- Dereck Lively: +15% usage → higher REB props

## Adding New Players

Edit `nba_ml/injuries.py` to add impact values:

```python
STAR_IMPACT = {
    # Your favorite player
    "Jimmy Butler": {"spread": 3.5, "total": 5.0, "usage_shift": 0.13},

    # Add usage beneficiaries in get_usage_boost_candidates()
    "MIA": {
        "Bam Adebayo": 1.20,
        "Tyler Herro": 1.25,
    },
}
```

## API Limitations

**ESPN API:**
- Free and reliable
- Sometimes delayed on late-scratches
- Best to double-check 1 hour before game time

**Manual Override:**
- Use when you have insider info
- For late scratches ESPN hasn't updated
- When you disagree with default impact values

## Tips for Betting with Injuries

1. **Check injury report 30-60 minutes before game** - late scratches are common
2. **Bigger edge on player props** - usage shifts can be 20-30% for remaining players
3. **Live betting opportunities** - if sportsbook is slow to adjust
4. **Stack teammate props** - when a star is out, multiple teammates benefit
5. **Fade the public** - casual bettors often overreact to star injuries

## Common Scenarios

### Scenario 1: Lakers without LeBron & AD
```bash
python -m nba_ml.predict --injuries="LAL:LeBron James,Anthony Davis"
```
- LAL spread: -9.0 pts worse
- Total: -11.0 pts lower
- **BET:** UNDER, opponent spread/ML

### Scenario 2: Warriors without Curry
```bash
python -m nba_ml.predict --injuries="GSW:Stephen Curry"
```
- GSW spread: -5.0 pts worse
- Total: -6.5 pts lower
- **PROPS:** Klay Thompson pts OVER, Draymond Green assists OVER

### Scenario 3: Bucks without Giannis
```bash
python -m nba_ml.predict --injuries="MIL:Giannis Antetokounmpo"
```
- MIL spread: -6.0 pts worse
- Total: -7.5 pts lower
- **PROPS:** Damian Lillard pts/ast OVER, Brook Lopez reb OVER

## Next Steps

1. Run predictions with current injuries
2. Compare adjusted lines to sportsbook lines
3. Identify edges (model adjusted more/less than book)
4. Target player props for teammates of injured stars
5. Track results and refine impact values over time
