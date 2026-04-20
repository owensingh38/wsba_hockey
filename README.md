# WSBA HOCKEY
![WSBA128](https://github.com/user-attachments/assets/4f349728-b99d-4e03-9d77-95cd177fefe2)

### A Python package for scraping and analyzing hockey data under the motto: ***Evaluating, analyzing, and understanding the game of hockey through the lens of different analytical methods, including incorporation of analytics.***

## INSTALLATION AND USAGE
```bash
pip install wsba_hockey
```

```python
import wsba_hockey as wsba
```

## ALL FEATURES

## SCRAPING
### NHL Play-by-Play (of any game frame up to a full season)
#### Functions:

```python
wsba.nhl_scrape_game(2024020918,split_shifts=False,remove=['game-end'])
wsba.nhl_scrape_season(20242025,split_shifts=False,remove=['game-end'],local=True)
```

### NHL Season Information

```python
wsba.nhl_scrape_schedule(20242025)
wsba.nhl_scrape_seasons_info(seasons=[20212022,20222023,20232024,20242025])
wsba.nhl_scrape_standings()
```

### NHL Rosters and Player Information

```python
wsba.nhl_scrape_roster(20242025)
wsba.nhl_scrape_player_info([8477956, 8479987])
wsba.nhl_scrape_team_info()
```

### NHL Draft Rankings and Prospects

```python
wsba.nhl_scrape_draft_rankings()
wsba.nhl_scrape_prospects('BOS')
```

### NHL EDGE Data
```python
wsba.nhl_scrape_edge(20252026,'skater',[8477956, 8479987])
wsba.nhl_scrape_edge(20252026,'goalie',[8480280])
wsba.nhl_scrape_edge(20252026,'team',['BOS'])
```

## DATA ANALYTICS
### Expected Goals
```python
pbp = wsba.nhl_scrape_game(2024020918,split_shifts=False,remove=['game-end'])
pbp = wsba.nhl_apply_xG(pbp)
```

### Goal Impacts and Shot Analysis

### Stat Aggregation
```python
pbp = wsba.nhl_scrape_season(20232024, local = True)
wsba.nhl_calculate_stats(pbp,'skater',['5v5','4v4','3v3'], 'all',shot_impact = True)
```
### Shot Plotting (Plots, Heatmaps, etc.)
```python
pbp = wsba.nhl_scrape_season(20212022, remove=[], local=True)

plots = wsba.nhl_plot_events(
    pbp,
    group="game",
    entities=[2021020002],
    events=["goal", "shot-on-goal", "missed-shot", "blocked-shot"],
    strengths=["5v5"],
    season_types=2,
    legend=True,
    rotation=0,
    display_range="full",
)
plots[2021020002].savefig("skater_events.png", bbox_inches="tight")
```
#### Sample Plot:
![Sample Plot](https://github.com/user-attachments/assets/62eef377-7f88-4f29-b6ba-2ca8d7810fa4)

## REPOSITORY 
### Team Information
```python
wsba.repo_load_teaminfo()
wsba.repo_load_rosters(seasons=[20212022,20222023,20232024,20242025])
```

### Schedule
```python
wsba.repo_load_schedule(seasons=[20212022,20222023,20232024,20242025])
```

## DOCUMENTATION
View full documentation here: [WSBA Hockey Package Documentation](https://weakside-breakout-analysis.github.io/wsba_hockey/)
