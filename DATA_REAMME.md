# Data Dictionary

## nba_historical_games.csv

Game-level data for NBA regular season and playoff games with team box score stats and betting lines.

| Column | Description |
|--------|-------------|
| `game_id` | Unique NBA game identifier |
| `date` | Game date (YYYY-MM-DD) |
| `season` | Season year (e.g., 1999 = 1999-00 season) |
| `home_team` | Home team abbreviation (e.g., NYK, LAL) |
| `away_team` | Away team abbreviation |
| `home_team_id` | NBA API team ID for the home team |
| `away_team_id` | NBA API team ID for the away team |
| `home_score` | Home team final score |
| `away_score` | Away team final score |
| `home_fg_made` | Home field goals made |
| `home_fg_att` | Home field goal attempts |
| `home_fg_pct` | Home field goal percentage |
| `home_fg3_made` | Home three-pointers made |
| `home_fg3_att` | Home three-point attempts |
| `home_fg3_pct` | Home three-point percentage |
| `home_ft_made` | Home free throws made |
| `home_ft_att` | Home free throw attempts |
| `home_ft_pct` | Home free throw percentage |
| `home_oreb` | Home offensive rebounds |
| `home_dreb` | Home defensive rebounds |
| `home_reb` | Home total rebounds |
| `home_ast` | Home assists |
| `home_stl` | Home steals |
| `home_blk` | Home blocks |
| `home_tov` | Home turnovers |
| `home_pf` | Home personal fouls |
| `away_fg_made` | Away field goals made |
| `away_fg_att` | Away field goal attempts |
| `away_fg_pct` | Away field goal percentage |
| `away_fg3_made` | Away three-pointers made |
| `away_fg3_att` | Away three-point attempts |
| `away_fg3_pct` | Away three-point percentage |
| `away_ft_made` | Away free throws made |
| `away_ft_att` | Away free throw attempts |
| `away_ft_pct` | Away free throw percentage |
| `away_oreb` | Away offensive rebounds |
| `away_dreb` | Away defensive rebounds |
| `away_reb` | Away total rebounds |
| `away_ast` | Away assists |
| `away_stl` | Away steals |
| `away_blk` | Away blocks |
| `away_tov` | Away turnovers |
| `away_pf` | Away personal fouls |
| `total_score` | Combined final score (home + away) |
| `home_margin` | Home team margin of victory (home_score - away_score) |
| `home_win` | 1 if home team won, 0 otherwise |
| `spread_line` | Betting spread line (negative = home favored) |
| `total_line` | Over/under betting line |
| `home_ml` | Home team moneyline odds |
| `away_ml` | Away team moneyline odds |
| `home_cover` | 1 if home team covered the spread |
| `total_over` | 1 if the game went over the total line |

---

## nba_player_game_logs.csv

Player-level box score stats for each game appearance.

| Column | Description |
|--------|-------------|
| `SEASON_ID` | Season identifier (e.g., 21999 = 1999-00 season) |
| `PLAYER_ID` | Unique NBA player ID |
| `PLAYER_NAME` | Player full name |
| `TEAM_ID` | NBA API team ID |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., CLE, NYK) |
| `GAME_ID` | Unique game identifier (joins to `nba_historical_games.game_id`) |
| `GAME_DATE` | Game date (YYYY-MM-DD) |
| `MIN` | Minutes played |
| `PTS` | Points scored |
| `REB` | Total rebounds |
| `AST` | Assists |
| `STL` | Steals |
| `BLK` | Blocks |
| `TOV` | Turnovers |
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage |
| `FG3M` | Three-pointers made |
| `FG3A` | Three-pointers attempted |
| `FG3_PCT` | Three-point percentage |
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage |
| `PLUS_MINUS` | Plus/minus while on court |

---

## Join Key

The two datasets can be joined on `nba_historical_games.game_id` = `nba_player_game_logs.GAME_ID` to connect player stats to game-level outcomes and betting lines.
