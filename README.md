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
### Note: Features yet included are marked with *


## SCRAPING
### NHL Play-by-Play (of any game frame up to a full season)
#### Functions:

```python
wsba.nhl_scrape_game(['2024020918'],split_shifts=False,remove=['game-end'])
wsba.nhl_scrape_season('20242025',split_shifts=False,remove=['game-end'],local=True)
```

### NHL Season Information

```python
wsba.nhl_scrape_schedule('20242025')
wsba.nhl_scrape_seasons_info(seasons=['20212022','20222023','20232024','20242025])
wsba.nhl_scrape_standings(arg = '2024-03-20')
```

### NHL Rosters and Player Information

```python
wsba.nhl_scrape_player_info(wsba.nhl_scrape_roster('20242025'))
```

## DATA ANALYTICS
### Expected Goals (WeakSide Breakout and MoneyPuck models)*
### Goal Impacts and Shot Analysis*
### Stat Aggregation*
### Shot Plotting (Plots, Heatmaps, etc.)*

## REPOSITORY 
### Past Season Play-by-Play*
### Team Information
```python
wsba.repo_load_teaminfo()
wsba.repo_load_rosters(seasons=['20212022','20222023','20232024','20242025])
```
### Schedule
```python
wsba.repo_load_schedule(seasons=['20212022','20222023','20232024','20242025])
```

## ACKNOWLEDGEMENTS AND CREDITS 
### Huge thanks to the following:
Harry Shomer - Creator of the hockey_scraper package, which contains select utils functions utilized in this package and otherwise inspires the creation of this package.

Dan Morse - Creator of the hockeyR package; another important inspiration and model for developing an NHL scraper.
