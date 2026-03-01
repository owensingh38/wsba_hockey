import os
import pandas as pd
import matplotlib.pyplot as plt
import wsba_hockey as wsba

### WSBA HOCKEY ###
## Provided below are some tests of package capabilities

dir = os.path.dirname(os.path.realpath(__file__))

#Test scrape of random games and exported rosters
pbp, rosters = wsba.nhl_scrape_game(['random',3,2007,2024], xg=True, export_roster=True)
pbp.to_csv(f'{dir}/samples/sample_random_pbp.csv',index=False)
rosters.to_csv(f'{dir}/samples/sample_random_game_rosters.csv',index=False)

#Retrieve skater game stats at 5v5
game_stats = wsba.nhl_calculate_stats(pbp,'skater','5v5')
game_stats.to_csv(f'{dir}/samples/sample_game_stats.csv',index=False)

#Find total sample stats and sort by goals
stats = wsba.nhl_agg_stats(game_stats, ['player_id'], sort={'goals': False}, comparison=False)
stats.to_csv(f'{dir}/samples/sample_agg_stats.csv',index=False)

#Plot xG shots in each game at even strength and then save the first to file
game = pbp['game_id'].unique()[0]
plots = wsba.nhl_plot_games(pbp, strengths=['5v5', '4v4', '3v3'])
plots[game].savefig(f'{dir}/samples/sample_game_plot.png', bbox_inches='tight')

#Standings Scraping
standings = wsba.nhl_scrape_standings(20222023)
standings.to_csv(f'{dir}/samples/sample_standings.csv',index=False)

#Get edge data for David Pastrnak and Morgan Geekie in 2025-26
edge_data = wsba.nhl_scrape_edge(20252026, 'skater', [8477956, 8479987])
edge_data.to_csv(f'{dir}/samples/sample_edge_data.csv',index=False)

#Collect schema of play-by-play dataframe
pbp_schema = wsba.utility_get_schema(pbp)
pbp_schema.to_csv(f'{dir}/samples/sample_schema.csv',index=False)