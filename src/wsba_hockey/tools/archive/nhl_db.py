import os
import pandas as pd
from typing import Literal, Union
from wsba_hockey.tools.scraping import *
from wsba_hockey.tools.xg_model import *
from wsba_hockey.tools.agg import *
from wsba_hockey.tools.plotting import *
from wsba_hockey.tools.columns import *
from wsba_hockey.tools.globals import *

## CLASSES ##
class NHL_Database:
    """
    A class for managing and analyzing NHL play-by-play data.

    This class supports game scraping, filtering, stat calculation, and plotting.
    It initializes with either a provided list of game IDs or a default/random set.

    Attributes:
        name (str):
            Designated name of the database.
        pbp (pd.DataFrame): 
            Combined play-by-play data for selected games.
        games (list[int]): 
            Unique game IDs currently in the dataset.
        stats (dict[str, dict[str, pd.DataFrame]]): 
            Dictionary storing calculated stats by type and name.
        plots (dict[int, matplotlib.figure.Figure] |  dict[str or int, dict[int, dict[str, matplotlib.figure.Figure]]]): 
            Dictionary storing plot outputs keyed by game or event.

    Args:
        game_ids (list[int], optional): 
            List of game IDs to scrape initially.
        pbp (pd.DataFrame, optional): 
            Existing PBP DataFrame to load instead of scraping.
    """

    def __init__(self, name:str, game_ids:list[int] = [], pbp:pd.DataFrame = pd.DataFrame()):
        """
        Initialize the WSBA_Database with scraped or preloaded PBP data.

        If no `pbp` is provided and `game_ids` is empty, a random set of games will be scraped.

        Args:
            name (str):
                Name of database.
            game_ids (list[int], optional): 
                List of NHL game IDs to scrape in initialization.
            pbp (pd.DataFrame, optional): 
                Existing play-by-play data to initialization.

        Returns:
            pd.DataFrame: 
                The initialized play-by-play dataset.
        """

        print(f'Initializing database "{name}"...')
        self.name = name

        if game_ids:
            self.pbp = nhl_apply_xG(nhl_scrape_game(game_ids))
        else:
            self.pbp = nhl_apply_xG(nhl_scrape_game(['random',3,2007,2024])) if pbp.empty else pbp

        self.games = self.pbp['game_id'].drop_duplicates().to_list()
        self.stats = {}
        self.game_plots = {}
        self.plots = {}

    def add_games(self, game_ids:list[int]):
        """
        Add additional games to the existing play-by-play dataset.

        Args:
            game_ids (list[int]): 
                List of game IDs to scrape and append.

        Returns:
            pd.DataFrame: 
                The updated play-by-play dataset.
        """

        print('Adding games...')
        self.pbp = pd.concat([self.pbp,nhl_apply_xG(nhl_scrape_game(game_ids))])

        return self.pbp
    
    def select_games(self, game_ids:list[int]):
        """
        Return a filtered subset of the PBP data for specific games.

        Args:
            game_ids (list[int]): 
                List of game IDs to include.

        Returns:
            pd.DataFrame: 
                Filtered PBP data matching the selected games.
        """
         
        print('Selecting games...')

        df = self.pbp
        return df.loc[df['game_id'].isin(game_ids)]

    def add_stats(self, name:str, type:Literal['skater','goalie','team'], game_strength:Union[Literal['all'], str, list[str]] = 'all', season_types:int | list[int] = 2, split_game:bool = False, roster_path:str = DEFAULT_ROSTER, shot_impact:bool = False):
        """
        Calculate and store statistics for the given play-by-play data.

        Args:
            name (str): 
                Key name to store the results under.
            type (Literal['skater', 'goalie', 'team']):
                Type of statistics to calculate. Must be one of 'skater', 'goalie', or 'team'.
            season (int): 
                The NHL season formatted such as "20242025".
            game_strength (int or list[str]):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            season_types (int or List[int], optional):
                List of season_types to include in scraping process.  Default is all regular season and playoff games which are 2 and 3 respectively.
            split_game (bool, optional):
                If True, aggregates stats separately for each game; otherwise, stats are aggregated across all games.  Default is False.
            roster_path (str, optional):
                File path to the roster data used for mapping players and teams.
            shot_impact (bool, optional):
                If True, applies shot impact metrics to the stats DataFrame.  Default is False.

        Returns:
            pd.DataFrame: 
                The calculated statistics.
        """

        df =  nhl_calculate_stats(self.pbp, type, game_strength, season_types, split_game, roster_path, shot_impact)
        self.stats.update({type:{name:df}})

        return df
    
    def get_players(self):
        """
        Return list of player IDs in the database.

        Returns:
            List: 
                List of player IDs.
        """

        return pd.unique(self.pbp[[
            'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id','away_goalie_id',
            'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id','home_goalie_id'
        ]].values.ravel()).tolist()
    
    def get_teams(self):
        """
        Return list of teams in the database.

        Returns:
            List: 
                List of teams IDs.
        """

        return pd.unique(self.pbp[['away_team_abbr','home_team_abbr']].values.ravel()).tolist()

    def get_seasons(self):
        """
        Return list of seasons in the database.

        Returns:
            List: 
                List of seasons IDs.
        """

        return pd.unique(self.pbp['season']).tolist()


    def add_game_plots(self, events:list[str] = FENWICK_EVENTS, strengths:Union[Literal['all'], list[str]] = 'all', game_ids: Union[Literal['all'], list[int]] = 'all', marker_dict:dict = event_markers, team_colors:dict = {'away':'primary','home':'primary'}, legend:bool = False):
        """
        Generate visualizations of game events based on play-by-play data.

        Args:
            events (list[str]):
                List of event types to include in the plot (e.g., ['shot-on-goal', 'goal']).
            strengths (str or list[str], optional):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            game_ids (str or list[int], optional):
                List of game IDs to plot. If set to 'all', plots will be generated for all games in the DataFrame.
            marker_dict (dict[str, dict], optional):
                Dictionary mapping event types to marker styles and/or colors used in plotting.
            team_colors (dict[str, str], optional):
                Dictionary mapping team venue (home or away) to its primary or secondary color.
            legend (bool, optional):
                Whether to include a legend on the plots.

        Returns:
            dict[int, matplotlib.figure.Figure]:
                A dictionary mapping each game ID to its corresponding matplotlib event plot figure.
        """
        
        self.game_plots.update(nhl_plot_games(self.pbp, events, strengths, game_ids, marker_dict, team_colors, legend))

        return self.game_plots
    
    def add_plots(self, plot:Literal['shot','heatmap'], player_dict:dict[str | int | Literal[8], list[int, str]], strengths:Union[Literal['all'], list[str]] = 'all', season_types:int | list[int] = 2, strengths_title:str | None = None, marker_dict:dict = event_markers, situation:Literal['indv','for','against'] = 'indv', titles:str | list[str] | None = None, legend:bool = False):
        """
        Generate visualizations for players or teams based on play-by-play data.

        Args:
            plot (str):
                Type of plot to generate (shot plot or heatmap)
            player_dict (dict[str, list[str]]):
                Dictionary of players to plot, where each key is a player name and the value is a list 
                with season and team info (e.g., {'Patrice Bergeron': [20212022, 'BOS']} or {8470638: [20212022, 'BOS']}).  
                Setting the key to the int value 8 will generate a heatmap for the full team.

                If generating a shot plot, only skaters can be plotted.
            strengths (str or list[str], optional):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            season_types (int or List[int], optional):
                List of season_types to include in scraping process.  Default is all regular season games which is the int '2'.
            strengths_title (str or None, optional):
                Specify a title to describe the strengths states included in the plot.  Default is None (strengths shown will be a full list of the included strengths in the plot).
            marker_dict (dict[str, dict]):
                Dictionary mapping event types to marker styles and/or colors used in plotting.  Only applies when plot is equal to 'shot'.
            situation (Literal['indv', 'for', 'against'], optional):
                Determines which shot events to include for the player:
                - 'indv': only the player's own shots,
                - 'for': shots taken by the player's team while they are on ice,
                - 'against': shots taken by the opposing team while the player is on ice.

                Only applies when plot is equal to 'shot'.
            titles (str or list[str], optional):
                List of titles for each plot defined in player_dict.  Use empty quotes for a blank title and if no titles argument is provided, use a default title.
            legend (bool):
                Whether to include a legend on the plots.  Only applies when plot is equal to 'shot'.

        Returns:
            Dict[str or int, Dict[int, Dict[str, matplotlib.figure.Figure]]]:
                A dictionary mapping each skater’s name or id to their corresponding season, team, then matplotlib heatmap figure.  The phrase 'Team' takes the place for team heatmaps.
        """
        
        data = nhl_plot_skaters_shots(self.pbp,player_dict,strengths,season_types,strengths_title,marker_dict,situation,titles,legend) if plot == 'shot' else nhl_plot_heatmap(self.pbp,player_dict,strengths,strengths_title,titles)

        self.plots.update(data)

        return self.plots    
    
    def export_data(self, path:str = ''):
        """
        Export the data within the object to a specified directory.

        The method writes:
        - The full play-by-play DataFrame to a CSV file.
        - All calculated statistics by type and name to CSV files in subfolders.
        - All stored plots to PNG files.

        If no path is provided, exports to a folder named after the database (`self.name/`).

        Args:
            path (str, optional): 
                Root folder to export data into. Defaults to `self.name/`.
        """

        print(f'Exporting data in database "{self.name}"...')
        start = time.perf_counter()

        # Use default path if none provided
        path = f'{self.name}/' if path == '' else os.path.join(path,f'{self.name}')
        os.makedirs(path, exist_ok=True)

        # Export master PBP
        self.pbp.to_csv(os.path.join(path, 'pbp.csv'), index=False)

        # Export stats
        for stat_type in self.stats.keys():
            for name, df in self.stats[stat_type].items():
                stat_path = os.path.join(path, 'stats', stat_type)
                os.makedirs(stat_path, exist_ok=True)
                df.to_csv(os.path.join(stat_path, f'{name}.csv'), index=False)

        # Export game plots
        plot_path = os.path.join(path, 'game_plots')
        os.makedirs(plot_path, exist_ok=True)
        for game_id, plot in self.game_plots.items():
            plot.savefig(os.path.join(plot_path, f'{game_id}.png'), bbox_inches='tight')

        # Export plots
        plot_path = os.path.join(path, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        for eid, seasons in self.plots.items():
            os.makedirs(f'{plot_path}/{eid}', exist_ok=True)
            for season, teams in seasons.items():
                os.makedirs(f'{plot_path}/{eid}/{season}', exist_ok=True)
                for team, plot in teams.items():
                    os.makedirs(f'{plot_path}/{eid}/{season}/{team}', exist_ok=True)
                    plot.savefig(os.path.join(plot_path, f'{eid}/{season}/{team}/plot.png'), bbox_inches='tight')

        # Completion message
        end = time.perf_counter()
        length = end - start
        print(f"...finished in {length:.2f} {'seconds' if length < 60 else 'minutes'}.")