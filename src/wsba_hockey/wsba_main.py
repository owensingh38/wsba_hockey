import os
import random
import matplotlib
import numpy as np
import pandas as pd
import matplotlib
import datetime as dt
import time
from numbers import Integral
from typing import Literal, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import JSONDecodeError
from wsba_hockey.tools.scraping import (
    combine_data,
    edge_stat_entry,
    get_game_info,
    parse_game_roster,
)
from wsba_hockey.tools.agg import (
    apply_params,
    apply_rosters,
    calc_goalie,
    calc_indv,
    calc_onice,
    calc_team,
    concat_col_values,
    extra_calc,
    rank_stats,
    sum_unique_games,
)
from wsba_hockey.tools.plotting import (
    apply_primary_colors,
    load_teaminfo,
    plot_events,
    team_primary_color_map,
)
from wsba_hockey.tools.xg_model import nhl_apply_xG
from wsba_hockey.tools.http import POOL_WORKERS, get as http_get, make_pooled_session
from wsba_hockey.tools.globals import (
    BIO_STAT_COL,
    COL_MAP,
    DEFAULT_AGG,
    DEFAULT_ROSTER,
    DRAFT_CAT,
    EVENTS,
    EVENT_MARKERS,
    FENWICK_EVENTS,
    FRONT_COL,
    INFO_PATH,
    KNOWN_PROBS,
    NON_TOTALS,
    SCHEDULE_PATH,
    SHOT_TYPES,
    STANDINGS_COLS,
    STATS_SORT,
    STRENGTHS,
    TEAMS,
)

### WSBA HOCKEY ###
## Provided below are all integral functions in the WSBA Hockey Python package. ##

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(
        game_ids:int | list[int], 
        split_shifts:bool = False,
        export_roster:bool = False,
        remove:list[str] = [], 
        xg:bool = False, 
        sources:bool = False,
        errors:bool = False
    ) -> Union[pd.DataFrame, 
               dict[str, pd.DataFrame],
               tuple[pd.DataFrame | dict[str, pd.DataFrame], pd.DataFrame]
        ]:
    """
    Given a set of game_ids (NHL API), return complete play-by-play information as requested.

    Args:
        game_ids (int or List[int] or ['random', int, int, int]):
            List of NHL game IDs to scrape or use ['random', n, start_year, end_year] to fetch n random games.
        split_shifts (bool, optional):
            If True, returns a dict with separate 'pbp' and 'shifts' DataFrames. Default is False.
        export_roster (bool, optional):
            If True, returns a second DataFrame with rosters for all players in the provided games. Default is False.
        remove (List[str], optional):
            List of event types to remove from the result. Default is an empty list.
        xg (bool, optional):
            If True, calculates xG for the play-by-play data (for most accurate values leave 'remove' empty).
        sources (bool, optional):
            If True, saves raw HTML, JSON, SHIFTS, and single-game full play-by-play to a separate folder in the working directory. Default is False.
        errors (bool, optional):
            If True, includes a list of game IDs that failed to scrape in the return. Default is False.

    Returns:
        If `split_shifts` is False, returns a single DataFrame of play-by-play data.

        If `split_shifts` is True, returns a dictionary with keys:

        - `'pbp'`: play-by-play events
        - `'shifts'`: shift change events
        - `'errors'` (optional): list of game IDs that failed if `errors=True`

        If `export_roster` is True, returns a tuple of (pbp, roster_df), where pbp is either a DataFrame
        or a dict (depending on `split_shifts`).
    """
    
    #Wrap game_id in a list if only a single game_id is provided
    game_ids = [game_ids] if type(game_ids) != list else game_ids

    pbps = []
    if game_ids[0] == 'random':
        #Randomize selection of game_ids
        #Some ids returned may be invalid (for example, 2020022000)
        num = game_ids[1]
        start = game_ids[2] if len(game_ids) > 1 else 2007
        end = game_ids[3] if len(game_ids) > 2 else (dt.date.today().year)-1

        game_ids = []
        i = 0
        print("Finding valid, random game ids...")
        while i is not num:
            print(f"\rGame IDs found in range {start}-{end}: {i}/{num}",end="")
            rand_year = random.randint(start,end)
            rand_season_type = random.randint(2,3)
            rand_game = random.randint(1,1312)

            #Ensure id validity (and that number of scraped games is equal to specified value)
            rand_id = f'{rand_year}{rand_season_type:02d}{rand_game:04d}'
            try: 
                #If game exists and has at least begun, then scraping can occur.
                rand_data = http_get(f"https://api-web.nhle.com/v1/gamecenter/{rand_id}/play-by-play").json()
                if rand_data['gameState'] == 'FUT':
                    continue
                else:
                    i += 1
                    game_ids.append(rand_id)
            except: 
                continue
        
        print(f"\rGame IDs found in range {start}-{end}: {i}/{num}")
            
    #Scrape each game
    #Track Errors
    rost_dfs = []
    error_ids = []
    prog = 0
    for game_id in game_ids:
        print(f'Scraping data from game {game_id}...',end='')
        start = time.perf_counter()

        try:
            #Retrieve data
            info = get_game_info(game_id)
            data = combine_data(info, sources)

            #Export roster if necessary
            roster = parse_game_roster(info['rosters'], game_id) if export_roster else None
            rost_dfs.append(roster)

            #Append data to list
            pbps.append(data)

            end = time.perf_counter()
            secs = end - start
            prog += 1
            
            #Export if sources is true
            if sources:
                source_season = info['season']
                source_game_id = info['game_id']
                dirs = f"sources/{source_season}/"

                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                data.to_csv(f"{dirs}{source_game_id}.csv",index=False)

            print(f" finished in {secs:.2f} seconds. {prog}/{len(game_ids)} ({(prog/len(game_ids))*100:.2f}%)")
        except Exception as e:
            #Games such as the all-star game and pre-season games will incur this error
            
            #Other games have known problems
            if game_id in KNOWN_PROBS.keys():
                print(f"\nGame {game_id} has a known problem: {KNOWN_PROBS[game_id]}")
            else:
                print(f"\nUnable to scrape game {game_id}.  Exception: {e}")
            
            #Track error
            error_ids.append(game_id)
            
    #Add all pbps together
    if not pbps:
        print("\rNo data returned.")
        return pd.DataFrame()
    df = pd.concat(pbps)
    rosters = pd.concat(rost_dfs) if export_roster else None

    #Add xG if necessary
    if xg:
        df = nhl_apply_xG(df)
    else:
        pass

    #Print final message
    if error_ids:
        print(f'\rScrape of provided games finished.\nThe following games failed to scrape: {error_ids}')
    else:
        print('\rScrape of provided games finished.')

    #Trim events as necessary and obtain play-by-play df
    pbp = df.loc[~df['event_type'].isin(remove)]
    
    #Split pbp and shift events if necessary
    #Return: complete play-by-play with data removed or split as necessary
    
    if split_shifts:
        pbp_dict = {
            "pbp": pbp.loc[~pbp['event_type'] != 'change'],
            "shifts": pbp.loc[df['event_type'] == 'change']
        }

        if errors:
            pbp_dict['errors'] = error_ids

        return (pbp_dict, rosters) if export_roster else pbp_dict

    else:
        if errors:
            pbp = {
                'pbp': pbp,
                'errors': error_ids
            }

        return (pbp, rosters) if export_roster else pbp

def nhl_scrape_schedule(
        season:int | Literal['now'] = 'now', 
        start:str | None = None, 
        end:str | None = None
    ) -> pd.DataFrame:
    """
    Given season and an optional date range, retrieve NHL schedule data.

    Args:
        season (int or str, optional): 
            The NHL season formatted such as "20242025" or "now".  Default is "now".
        start (str, optional): 
            The date string (MM-DD) to start the schedule scrape at. Default is None
        end (str, optional): 
            The date string (MM-DD) to end the schedule scrape at. Default is None

    Returns:
        pd.DataFrame: 
            A DataFrame containing the schedule data for the specified season and date range.
    """

    api = "https://api-web.nhle.com/v1/score/"
    form = '%Y-%m-%d'

    #If the season argument is now (live schedule) then skip this step
    if season == 'now':
        #Set start and end to filler values to ensure only one date is scraped (the phrase 'now' will be appened pre-scrape)
        start = end = dt.datetime.now()
    else:
        season_data = http_get('https://api.nhle.com/stats/rest/en/season').json()['data']
        season_data = [s for s in season_data if s['id'] == int(season)][0]

        #Select start and end dates for scrape (if none are provided use the official season start and end dates)
        #Determine how to approach scraping; if month in season is after the new year the year must be adjusted
        season_start = f'{(str(season)[0:4] if int(start[0:2])>=9 else str(season)[4:8])}-{start[0:2]}-{start[3:5]}' if start else season_data['startDate'][0:10]
        season_end = f'{(str(season)[0:4] if int(end[0:2])>=9 else str(season)[4:8])}-{end[0:2]}-{end[3:5]}' if end else season_data['endDate'][0:10]

        #Create datetime values from dates
        start = dt.datetime.strptime(season_start,form)
        end = dt.datetime.strptime(season_end,form)

    game = []

    day = (end-start).days+1
    if day < 0:
        #Handles dates which are over a year apart
        day = 365 + day
    for i in range(day):
        now = season == 'now'
        inc = start+dt.timedelta(days=i)
        
        date_string = 'now' if now else str(inc)[:10]

        #For each day, call NHL api and retreive info on all games of selected game
        date_context = 'as of' if now else 'on'
        print(f'Scraping games {date_context} {date_string}...')
        
        get = http_get(f'{api}{date_string}').json()
        game_week = pd.json_normalize(get['games']).drop(columns=['goals'],errors='ignore')
        
        #Return nothing if there's nothing
        if game_week.empty:
            game.append(game_week)
        else:
            game_week['game_date'] = get['currentDate']
            game_week['game_title'] = game_week['awayTeam.abbrev'] + " @ " + game_week['homeTeam.abbrev'] + " - " + game_week['game_date']
            game_week['start_time_est'] = pd.to_datetime(game_week['startTimeUTC']).dt.tz_convert('US/Eastern').dt.strftime("%I:%M %p")

        game.append(game_week)
        
    #Concatenate all games and standardize column naming
    df = pd.concat(game).rename(columns=COL_MAP['schedule'],errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]

    #Set logo links to dark variants (if any data exists)
    try:
        for team in ['away','home']:
            df[f'{team}_team_logo'] = df[f'{team}_team_logo'].str.replace('light','dark')
    except KeyError:
        print('No games found for range of dates provided.')

    #Return: specificed schedule data
    return df[[col for col in COL_MAP['schedule'].values() if col in df.columns]]

def nhl_scrape_season(
        season:int, 
        split_shifts:bool = False,
        export_roster:bool = False,
        season_types:list[int] = [2,3], 
        remove:list[str] = [], 
        start:str | None = None, 
        end:str | None = None, 
        local:bool=False, local_path:str = SCHEDULE_PATH, 
        xg:bool = False, 
        sources:bool = False, 
        errors:bool = False
    ) -> Union[pd.DataFrame, 
               dict[str, pd.DataFrame],
               tuple[pd.DataFrame | dict[str, pd.DataFrame], pd.DataFrame]
        ]:
    """
    Given season, scrape all play-by-play occuring within the season.

    Args:
        season (int): 
            The NHL season formatted such as "20242025".
        split_shifts (bool, optional):
            If True, returns a dict with separate 'pbp' and 'shifts' DataFrames. Default is False.
        export_roster (bool, optional):
            If True, returns a second DataFrame with rosters for all players in the provided games. Default is False.
        season_types (List[int], optional):
            List of season_types to include in scraping process.  Default is all regular season and playoff games which are 2 and 3 respectively.
        remove (List[str], optional):
            List of event types to remove from the result. Default is an empty list.
        start (str, optional): 
            The date string (MM-DD) to start the schedule scrape at. Default is None
        end (str, optional): 
            The date string (MM-DD) to end the schedule scrape at. Default is None
        local (bool, optional):
            If True, use local file to retreive schedule data.
        local_path (bool, optional):
            If True, specifies the path with schedule data necessary to scrape a season's games (only relevant if local = True).
        xg (bool, optional):
            If True, calculates xG for the play-by-play data (for most accurate values leave 'remove' empty).
        sources (bool, optional):
            If True, saves raw HTML, JSON, SHIFTS, and single-game full play-by-play to a separate folder in the working directory. Default is False.
        errors (bool, optional):
            If True, includes a list of game IDs that failed to scrape in the return. Default is False.

    Returns:
        If `split_shifts` is False, returns a single DataFrame of play-by-play data.

        If `split_shifts` is True, returns a dictionary with keys:

        - `'pbp'`: play-by-play events
        - `'shifts'`: shift change events
        - `'errors'` (optional): list of game IDs that failed if `errors=True`
    """
     
    #Determine whether to use schedule data in repository or to scrape
    local_failed = False

    if local:
        try:
            load = pd.read_csv(local_path)
            load['game_date'] = pd.to_datetime(load['game_date'])
            
            season_data = http_get('https://api.nhle.com/stats/rest/en/season').json()['data']
            season_data = [s for s in season_data if s['id'] == season][0]

            season_start = f'{(str(season)[0:4] if int(start[0:2])>=9 else str(season)[4:8])}-{start[0:2]}-{start[3:5]}' if start else season_data['startDate'][0:10]
            season_end = f'{(str(season)[0:4] if int(end[0:2])>=9 else str(season)[4:8])}-{end[0:2]}-{end[3:5]}' if end else season_data['endDate'][0:10]

            #Create datetime values from dates
            start_date = pd.to_datetime(season_start)
            end_date = pd.to_datetime(season_end)

            load = load.loc[(load['season']==season)&
                            (load['season_type'].isin(season_types))&
                            (load['game_date']>=start_date)&(load['game_date']<=end_date)&
                            (load['game_schedule_state']=='OK')&
                            (load['game_state']!='FUT')
                            ]
            
            game_ids = load['game_id'].to_list()
        except KeyError:
            #If loading games locally fails then force a scrape
            local_failed = True
            print('Loading games locally has failed.  Loading schedule data with a scrape...')
    else:
        local_failed = True

    if local_failed:
        load = nhl_scrape_schedule(season,start,end)
        load = load.loc[(load['season']==season)&
                        (load['season_type'].isin(season_types))&
                        (load['game_schedule_state']=='OK')&
                        (load['game_state']!='FUT')
                        ]
        
        game_ids = load['game_id'].to_list()

    #If no games found, terminate the process
    if not game_ids:
        print('No games found for dates in season...')
        return ""
    
    print(f"Scraping games from {str(season)[0:4]}-{str(season)[4:8]} season...")
    start = time.perf_counter()

    #Perform scrape
    data = nhl_scrape_game(game_ids,split_shifts,export_roster,remove=remove,xg=xg,sources=sources,errors=errors)
    
    end = time.perf_counter()
    secs = end - start
    
    print(f'Finished season scrape in {(secs/60)/60:.2f} hours.')
    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    return data

def nhl_scrape_seasons_info(seasons:list[int] = []) -> pd.DataFrame:
    """
    Returns info related to NHL seasons (by default, all seasons are included)
    Args:
        seasons (List[int], optional): 
            The NHL season formatted such as "20242025".

    Returns:
        pd.DataFrame: 
            A DataFrame containing the information for requested seasons.
    """

    print(f'Scraping info for seasons: {seasons}')
    
    #Load two different data sources: general season info and standings data related to season
    api = "https://api.nhle.com/stats/rest/en/season"
    info = "https://api-web.nhle.com/v1/standings-season"
    data = http_get(api).json()['data']
    data_2 = http_get(info).json()['seasons']

    df = pd.json_normalize(data)
    df_2 = pd.json_normalize(data_2)

    #Remove common columns
    df_2 = df_2.drop(columns=['conferencesInUse', 'divisionsInUse', 'pointForOTlossInUse','rowInUse','tiesInUse','wildcardInUse'])
    
    df = pd.merge(df,df_2,how='outer',on=['id']).rename(columns=COL_MAP['season_info'])
    
    df = df[[col for col in COL_MAP['season_info'].values() if col in df.columns]]

    if len(seasons) > 0:
        return df.loc[df['season'].isin(seasons)].sort_values(by=['season'])
    else:
        return df.sort_values(by=['season'])

def nhl_scrape_standings(arg:int | list[int] | Literal['now'] = 'now', season_type:int = 2) -> pd.DataFrame:
    """
    Returns standings or playoff bracket
    Args:
        arg (int or list[int] or str, optional):
            Date formatted as 'YYYY-MM-DD' to scrape standings, NHL season such as "20242025", list of NHL seasons, or 'now' for current standings. Default is 'now'.
        season_type (int, optional):
            Part of season to scrape.  If 3 (playoffs) then scrape the playoff bracket for the season implied by arg. When arg = 'now' this is defaulted to the most recent playoff year.  Any dates passed through are parsed as seasons. Default is 2.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the standings information (or playoff bracket).
    """

    current_year = dt.datetime.now().year

    if season_type == 3:
        if arg == "now":
            arg = [current_year]
        elif type(arg) == int:
            #Find year from season
            arg = [str(arg)[4:8]]
        elif type(arg) == list:
            #Find year from seasons
            arg = [str(s)[4:8] for s in arg]
        else:
            #Find year from season from date
            arg = [int(arg[0:4])+1 if (9 < int(arg[5:7]) < 13) else int(arg[0:4])]

        print(f"Scraping playoff bracket for season{'s' if len(arg)>1 else ''}: {arg}")
        
        dfs = []
        for season in arg:
            api = f"https://api-web.nhle.com/v1/playoff-bracket/{season}"

            try:
                data = http_get(api).json()['series']
                dfs.append(pd.json_normalize(data))
            except Exception as e:
                print(f"Error scraping playoff bracket for season {season}: {e}")
                dfs.append(pd.DataFrame())

        #Combine and standardize columns
        df = pd.concat(dfs).rename(columns=COL_MAP['standings'])

        #Return: playoff bracket
        return df[[col for col in COL_MAP['standings'].values() if col in df.columns]]

    else:
        if arg == "now":
            print("Scraping standings as of now...")
            arg = [arg]
        elif arg in nhl_scrape_seasons():
            print(f'Scraping standings for season: {arg}')
            arg = [arg]
        elif type(arg) == list:
            print(f'Scraping standings for seasons: {arg}')
        else:
            print(f"Scraping standings for date: {arg}")
            arg = [arg]

        dfs = []
        for search in arg:
            #If the end is an int then its a season otherwise it is either 'now' or a date as a string
            if type(search) == int:
                #Check if the season date is during the requested season - if so then use this date to find the current standings for the requested season
                season_data = http_get('https://api.nhle.com/stats/rest/en/season').json()['data']
                season_data = [s for s in season_data if s['id'] == search][0]
                
                season_start = season_data['startDate']
                season_end = season_data['regularSeasonEndDate']

                today = dt.datetime.now().strftime("%Y-%m-%d")
                
                if season_start <= today <= season_end:
                    end = today[0:10]
                else:
                    end = season_end[0:10]
            else:
                end = search
                
            api = f"https://api-web.nhle.com/v1/standings/{end}"

            try:
                data = http_get(api).json()['standings']
                dfs.append(pd.json_normalize(data))
            except Exception as e:
                print(f"Error scraping standings for date {end}: {e}")
                dfs.append(pd.DataFrame())

        #Standardize columns
        df = pd.concat(dfs).rename(columns=COL_MAP['standings'])
        
        df['wsba_id'] = df['team_abbr'].astype(str) + df['season'].astype(str)
        
        #Return: standings data
        return df[[col for col in COL_MAP['standings'].values() if col in df.columns]]

def nhl_scrape_game_roster(game_ids: int | list[int]) -> pd.DataFrame:
    """
    Returns rosters for a list of individual games

    Args:
        game_ids (int or List[int] or ['random', int, int, int]):
            List of NHL game IDs to scrape or use ['random', n, start_year, end_year] to fetch n random games.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the rosters for all games in the specified list.
    """
    #Wrap game_id in a list if only a single game_id is provided
    game_ids = [game_ids] if type(game_ids) != list else game_ids

    #Prepare session to speed up requests
    global session
    session = make_pooled_session()

    #Helper function to quickly retrieve roster
    def fetch_roster(game, rest):
        print(f'Scraping rosters for game {game}...')

        r = session.get(f'https://api-web.nhle.com/v1/gamecenter/{game}/play-by-play', timeout=10)
        data = r.json()
        roster = parse_game_roster(pd.json_normalize(data['rosterSpots']), game)

        time.sleep(rest)
        return roster

    #Re-use the game info for pbp to just get the roster
    dfs = []
    errors = []

    dfs = []
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as ex:
        futures = {ex.submit(fetch_roster, g, 2): g for g in game_ids}
        for f in as_completed(futures):
            g = futures[f]
            try:
                r = f.result()
                if r is not None:
                    dfs.append(r)
            except Exception as e:
                print(f"\nUnable to scrape game {g}.  Exception: {e}")
                errors.append(g)

    if dfs:
        rosters = pd.concat(dfs, ignore_index=True)
    else:
        print("\rNo data returned.")
        return pd.DataFrame()

    #Add team abbreviations
    rosters['team_abbr'] = rosters['team_id'].replace(TEAMS)

    #Print final message
    if errors:
        print(f'\rScrape of provided games finished.\nThe following games failed to scrape: {errors}')
    else:
        print('\rScrape of provided games finished.')

    #Return: roster data for provided games
    return rosters

def nhl_scrape_roster(season: int, teams: str | list[str] | None = None) -> pd.DataFrame:
    """
    Returns rosters for a selection teams in a given season.

    Args:
        season (int):
            The NHL season formatted such as "20242025".

        teams (str or list[str], optional):
            List of teams(three letter abbreviation) to scrape.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the rosters for all teams in the specified season.
    """

    print(f'Scrpaing rosters for the {season} season...')
    teaminfo = pd.read_csv(INFO_PATH)

    if isinstance(teams, str):
        teams = [teams]
    elif not teams:
        teams = teaminfo['team_abbr'].drop_duplicates()

    rosts = []
    for team in teams:
        try:
            print(f'Scraping {team} roster...')
            api = f'https://api-web.nhle.com/v1/roster/{team}/{season}'
            
            data = http_get(api).json()
            forwards = pd.json_normalize(data['forwards'])
            forwards['heading_position'] = "F"
            dmen = pd.json_normalize(data['defensemen'])
            dmen['heading_position'] = "D"
            goalies = pd.json_normalize(data['goalies'])
            goalies['heading_position'] = "G"

            roster = pd.concat([forwards,dmen,goalies]).reset_index(drop=True)
            roster['player_name'] = (roster['firstName.default']+" "+roster['lastName.default']).str.upper()
            roster['season'] = season
            roster['team_abbr'] = team

            rosts.append(roster)
        except:
            print(f'No roster found for {team}...')
            rosts.append(pd.DataFrame())

    #Combine rosters
    df = pd.concat(rosts)

    #Standardize columns
    df = df.rename(columns=COL_MAP['roster'])

    #Return: roster data for provided season
    return df[[col for col in COL_MAP['roster'].values() if col in df.columns]]

def nhl_scrape_prospects(team:str) -> pd.DataFrame:
    """
    Returns prospects for specified team

    Args:
        team (str):
            Three character team abbreviation such as 'BOS'

    Returns:
        pd.DataFrame: 
            A DataFrame containing the prospect data for the specified team.
    """

    api = f'https://api-web.nhle.com/v1/prospects/{team}'

    data = http_get(api).json()

    print(f'Scraping {team} prospects...')

    #Iterate through positions
    players = [pd.json_normalize(data[pos]) for pos in ['forwards','defensemen','goalies']]

    prospects = pd.concat(players)
    #Add name columns
    prospects['player_name'] = (prospects['firstName.default']+" "+prospects['lastName.default']).str.upper()

    #Standardize columns
    prospects = prospects.rename(columns=COL_MAP['prospects'])
    
    #Return: team prospects
    return prospects[[col for col in COL_MAP['prospects'].values() if col in prospects.columns]]

def nhl_scrape_team_info(country:bool = False) -> pd.DataFrame:
    """
    Returns team or country information from the NHL API.

    Args:
        country (bool, optional):
            If True, returns country information instead of NHL team information.

    Returns:
        pd.DataFrame: 
            A DataFrame containing team or country information from the NHL API.
    """

    info_type = 'country' if country else 'team'
    print(f'Scraping {info_type} information...')
    api = f'https://api.nhle.com/stats/rest/en/{info_type}'
    
    data =  pd.json_normalize(http_get(api).json()['data'])

    #Add logos if necessary
    if not country:
        data['logo_light'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_light.svg'
        data['logo_dark'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_dark.svg'

    #Standardize columns
    data = data.rename(columns=COL_MAP['team_info'])

    #Return: team or country info 
    return data[[col for col in COL_MAP['team_info'].values() if col in data.columns]].sort_values(by=(['country_abbr','country_name'] if country else ['team_abbr','team_name']))

def nhl_scrape_player_info(player_ids: list[int]) -> pd.DataFrame:
    """
    Returns player data for specified players.

    Args:
        player_ids (list[int]):
            List of NHL API player IDs to retrieve information for.

    Returns:
        pd.DataFrame: 
            A DataFrame containing player data for specified players.
    """

    print(f'Retreiving player information for {player_ids}...')

    #Wrap game_id in a list if only a single game_id is provided
    player_ids = [player_ids] if type(player_ids) != list else player_ids

    session = make_pooled_session()

    def fetch_player(player_id, rest):
        player_id = int(player_id)
        api = f'https://api-web.nhle.com/v1/player/{player_id}/landing'

        try:
            data = pd.json_normalize(session.get(api, timeout=10).json())
            #Add name column
            data['player_name'] = (data['firstName.default'] + " " + data['lastName.default']).str.upper()
            time.sleep(rest)
            return data
        except JSONDecodeError:
            return None

    infos = []
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as executor:
        infos = list(executor.map(lambda pid: fetch_player(pid, 1), player_ids))

    session.close()

    infos = [df for df in infos if df is not None and not df.empty]
    if infos:
        df = pd.concat(infos)
        
        #Standardize columns
        df = df.rename(columns=COL_MAP['player_info'])

        #Return: player data
        return df[[col for col in COL_MAP['player_info'].values() if col in df.columns]]
    else:
        return pd.DataFrame()

def nhl_scrape_draft_rankings(arg:str | Literal['now'] = 'now', category:int = 0) -> pd.DataFrame:
    """
    Returns draft rankings
    Args:
        arg (str, optional):
            Date formatted as 'YYYY-MM-DD' to scrape draft rankings for specific date or 'now' for current draft rankings. Default is 'now'.
        category (int, optional):
            Category number for prospects. When ``arg='now'`` this does not apply. Categories: 1=North American Skaters, 2=International Skaters, 3=North American Goalies, 4=International Goalies. Default is 0 (all prospects).
            
    Returns:
        pd.DataFrame: 
            A DataFrame containing draft rankings.
    """

    print(f'Scraping draft rankings for {arg}...\nCategory: {DRAFT_CAT[category]}...')

    #Player category only applies when requesting a specific season
    api = f"https://api-web.nhle.com/v1/draft/rankings/{arg}/{category}" if category > 0 else f"https://api-web.nhle.com/v1/draft/rankings/{arg}"
    data = pd.json_normalize(http_get(api).json()['rankings'])

    #Add player name columns
    data['player_name'] = (data['firstName']+" "+data['lastName']).str.upper()

    #Fix positions
    data['positionCode'] = data['positionCode'].replace({
        'LW':'L',
        'RW':'R'
    })

    #Standardize columns
    data = data.rename(columns=COL_MAP['draft_rankings'])

    #Return: prospect rankings
    return data[[col for col in COL_MAP['draft_rankings'].values() if col in data.columns]]

def nhl_scrape_game_info(game_ids:list[int]) -> pd.DataFrame:
    """
    Given a set of game_ids (NHL API), return information for each game.

    Args:
        game_ids (List[int] or ['random', int, int, int]):
            List of NHL game IDs to scrape or use ['random', n, start_year, end_year] to fetch n random games.
    
    Returns:
        pd.DataFrame:
            An DataFrame containing information for each game.    
    """

    #Wrap game_id in a list if only a single game_id is provided
    game_ids = [game_ids] if type(game_ids) != list else game_ids

    print(f'Finding game information for games: {game_ids}')

    link = 'https://api-web.nhle.com/v1/gamecenter'

    #Scrape information
    df = pd.concat([pd.json_normalize(http_get(f'{link}/{game_id}/landing').json()) for game_id in game_ids])

    #Add extra info
    df['game_date'] = df['gameDate']
    df['game_title'] = df['awayTeam.abbrev'] + " @ " + df['homeTeam.abbrev'] + " - " + df['game_date']
    df['start_time_est'] = pd.to_datetime(df['startTimeUTC']).dt.tz_convert('US/Eastern').dt.strftime("%I:%M %p")

    #Standardize columns
    df = df.rename(columns=COL_MAP['schedule'])

    #Return: game information
    return df[[col for col in COL_MAP['schedule'].values() if col in df.columns]]

def nhl_scrape_edge(season: int, group: Literal['skater','goalie','team'], scrape: list[int | str], season_type:int = 2) -> pd.DataFrame:
    """
    Returns NHL Edge stats and data for a selection of skaters, goalies, or teams in a given season.

    Args:
        season (int):
            The NHL season formatted such as "20242025".
        group (Literal['skater', 'goalie', 'team']):
            Type of statistics to calculate. Must be one of 'skater', 'goalie', or 'team'.
        scrape (list[int or str]):
            List of skaters, goalies, or teams to scrape (player_ids for skaters/goalies and three letter abbreviation (i.e. 'BOS') for teams.)
        season_types (int, optional):
            Season type to include in scraping process.  Default is all regular season games which is the int '2'.

    Returns:
        pd.DataFrame:
            A DataFrame containing NHL EDGE metrics for the requested
            skaters, goalies, and/or teams for the specified season.
    """
    
    print(f'Scraping {group} edge data for the {season} season...')
    start = time.perf_counter()

    #NHL edge endpoint for teams uses their team ID rather than their three-letter abbreviation
    if group == 'team':
        data = nhl_scrape_team_info()
        teams = data.set_index('team_abbr')['team_id'].to_dict()

        print(teams)

        entries = [teams[team] for team in scrape]
    else:
        entries = scrape

    #EDGE data consists of the following categories:
    #Distance
    #Speed
    #Zone Time
    #Shot Speed
    #Shot Location

    #Iterate through each category and merge the total df to create a full EDGE stats df
    dfs = []
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as executor:
        futures = {executor.submit(edge_stat_entry, entry, season, season_type, group): entry for entry in entries}
        for future in as_completed(futures):
            try:
                df_entry = future.result()
            except Exception as e:
                print(f'Error fetching {futures[future]}: {e}')
                df_entry = pd.DataFrame()

            if not df_entry.empty:
                dfs.append(df_entry)

    if not dfs:
        return pd.DataFrame()

    #Combine edge data
    df = pd.concat(dfs, ignore_index=True)

    end = time.perf_counter()
    length = end-start
    elapsed = length if length < 60 else length / 60
    elapsed_unit = 'seconds' if length < 60 else 'minutes'
    print(f'...finished in {elapsed:.2f} {elapsed_unit}.')

    if df.empty:
        return df
    else:
        #Standardize columns
        df = df.rename(columns=COL_MAP['edge'])

        #Add additional columns
        df['season_type'] = season_type
        df['wsba_id'] = df['team_abbr']+df['season'].astype(str) if group == 'team' else df['player_id'].astype(str)+df['season'].astype(str)+df['team_abbr']

        #Return: dataframe including NHL Edge data for the specified group and the entries included
        return df[[col for col in COL_MAP['edge'].values() if col in df.columns]]

def nhl_scrape_seasons(analytic: bool = False) -> pd.DataFrame:
    """
    Returns list of NHL seasons

    Args:
        analytic (bool, optional):
            Filters list of seasons to those only included in the WSBA Hockey package (2007-2008 and beyond) if True.  Default is False.

    Returns:
        pd.DataFrame:
            A DataFrame containing a list of all NHL seasons.
    """

    data = http_get('https://api-web.nhle.com/v1/season').json()

    if analytic:
        data = [season for season in data if season > 20062007]

    return data


def nhl_calculate_stats(
        pbp:pd.DataFrame, 
        group:Literal['skater','goalie','team','game_score'], 
        game_strength:Union[Literal['all'], str, list[str]] = 'all', 
        season_types:int | list[int] = [2,3],
        schedule_path:str = SCHEDULE_PATH,
        roster_path:str = DEFAULT_ROSTER
    ) -> pd.DataFrame:
    """
    Given play-by-play data, seasonal information, game strength, rosters, and an xG model,
    return raw-total statistics at the game level for skaters, goalies, or teams.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data.
        group (Literal['skater', 'goalie', 'team', 'game_score']):
            Type of statistics to calculate. Must be one of 'skater', 'goalie', 'team', or 'game_score' (specific combination of skaters and goaltenders by game).
        season (int): 
            The NHL season formatted such as "20242025".
        game_strength (int or list[str], optional):
            List of game strength states to include (e.g., ['5v5','5v4','4v5']).  Default is 'all'.
        season_types (int or List[int], optional):
            List of season_types to include in scraping process.  Default is all regular season and playoff games which are the integers 2 and 3 respectively.
        split_game (bool, optional):
            If True, aggregates stats separately for each game; otherwise, stats are aggregated across all games.  Value is ignored when group == 'game_score'.  Default is False.
        roster_path (str, optional):
            File path to the roster data used for mapping players and teams.
            
    Returns:
        pd.DataFrame:
            A DataFrame containing the aggregated statistics according to the selected parameters.
    """

    seasons = pbp['season'].drop_duplicates().dropna().astype(int)
   
    season_type_label = (
        'regular season' if season_types == 2 else
        'playoff' if season_types == 3 else
        'regular season and playoff' if season_types == [2, 3] else
        'unknown selection of'
    )
    print(f'''Calculating statistics for {season_type_label} games in the provided play-by-play data at {game_strength} for {group}s...\nSeasons included: {seasons.to_list()}...'''
    )
    start = time.perf_counter()

    #Check if xG column exists and apply model if it does not
    if 'xG' not in pbp.columns:
        print('Applying xG model...')
        pbp = nhl_apply_xG(pbp)
    
    #If single values provided for columns typically in a list then place them into a list
    if isinstance(season_types, int):
        season_types = [season_types]
    if isinstance(game_strength, str) and game_strength != 'all':
        game_strength = [game_strength]

    #Apply season_type filter, remove shootouts, and remove invalid strengths
    pbp = pbp.loc[(pbp['season_type'].isin(season_types))&(pbp['period_type']!='SO')&(pbp['strength_state'].isin(STRENGTHS))]

    #Convert all columns with player ids to float in order to avoid merging errors
    id_cols = [col for col in pbp.columns if '_id' in col]
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(pbp[col]):
            pbp[col] = pd.to_numeric(pbp[col], errors='coerce')

    second_group = ['season', 'game_id']

    #Split calculation
    if group == 'goalie':
        complete = calc_goalie(pbp,game_strength,second_group)

    elif group == 'team':
        complete = calc_team(pbp,game_strength,second_group)  

    else:
        indv_stats = calc_indv(pbp,game_strength,second_group)
        onice_stats = calc_onice(pbp,game_strength,second_group)

        #IDs sometimes set as objects
        indv_stats['player_id'] = indv_stats['player_id'].astype(float)
        onice_stats['player_id'] = onice_stats['player_id'].astype(float)

        merge_cols = ['player_id','team_abbr'] + second_group
        complete = pd.merge(indv_stats, onice_stats, how='outer', on=merge_cols)
    
    #Remove entries with no ID listed
    if 'player_id' in complete.columns:
        complete = complete.loc[complete['player_id'].notna()]

    #Remove entries with no time on ice
    complete = complete.loc[complete['time_on_ice'].notna()]

    #Set TOI to minutes
    complete['time_on_ice'] = complete['time_on_ice'] / 60

    if group != 'game_score':
        complete['time_on_ice_per_games_played'] = complete['time_on_ice']/complete['games_played'] 

    #Apply roster information to stats
    sort_info = STATS_SORT[group]
    
    complete = apply_rosters(complete, group, schedule_path, roster_path).fillna(0).sort_values(by=sort_info['by'], ascending=sort_info['ascending'])

    #Add strength and season type columns to the end of the df
    complete['strength_state'] = game_strength if isinstance(game_strength, str) else ', '.join(game_strength)
    complete['season_type'] = 'all' if season_types == [2,3] else season_types if isinstance(season_types, int) else ', '.join([str(s) for s in season_types])
    
    end = time.perf_counter()
    length = end-start
    elapsed = length if length < 60 else length / 60
    elapsed_unit = 'seconds' if length < 60 else 'minutes'
    print(f'...finished in {elapsed:.2f} {elapsed_unit}.')

    return complete

def nhl_agg_stats(
        games_df:pd.DataFrame, 
        group_by:list[Literal['player_id','season','team_abbr','position','season_type','strength_state']] = DEFAULT_AGG, 
        params:dict | None = None, 
        sort:dict = {}, 
        metrics:list[tuple] = [], 
        rates:bool = True,
        comparison:bool = True, 
        exclude:list = [],
        manual_agg:dict[str] = {},
        schedule_path:str = SCHEDULE_PATH, 
        roster_path:str | None = None
    ) -> pd.DataFrame:
    """
    Given statistical data, columns, and rosters,
    return aggregated statistics at the skater, goalie, or team level.

    Args:
        games_df (pd.DataFrame):
            A DataFrame already containing game-by-game statistical data (generated with nhl_calculate_stats).
        group_by (list[str], optional):
            List of columns to group by.  You may provide an optional unspecified but this is currently unstable.
        params (dict or None, optional):
            Parameters to filter the games_df by before aggregating.  Default is None.
            In order to filter correctly, set each key to the desired column name in the dataframe and the value to the expression to filter by.
            A third element in the tuple value can indicate whether to perform the filter before aggregating or after.  By default, it will occur before (using 'before' or 'after').

            Ex. 'TOI': ('>=', 150, 'before') or 'Date': ('between', '2025-12-01', '2026-01-01')
        sort (dict[str], optional):
            Dict of values formatted with the sort column as the key and a bool determining whether to sort ascending or not as the value.  Default is empty leading to default sort.
        metric (list[tuple], optional):
            List of additional metrics to calculate.
            Use one of '+', '-', '*', '/' to perform an operation on any existing column (using pd.eval).
            The first tuple element should be the name of the metric value, the second the numerator, and the third should be the denominator (if there is none then pass None).

            Ex. [('time_on_ice_per_games_played', 'time_on_ice', 'games_played'), ('goals_saved_above_expected', 'expected_goals_against-goals_against', None)]
        comparison (bool, optional):
            If True, calculates rate (per-sixty minutes of ice time) stats.  Default is True.
        comparison (bool, optional):
            If True, calculates percentiles for all (applicable) numeric values in the dataframe.  Default is True.
        exclude (list[str], optional):
            List of columns to exclude from summation.  Default is None (summing all columns that are not grouped by).
        manual_agg (dict[str], optional):
            Dict with manual aggregation clause.  Default is empty dict.
        schedule_path (bool, optional):
            If True, specifies the path with schedule data necessary to add schedule data to games_df.
        roster_path (str or None, optional):
            File path to the roster data used for mapping players and teams.

    Returns:
        pd.DataFrame:
            A DataFrame containing the aggregated statistics according to the selected parameters.
    """
    #Seasons list
    try:
        seasons = games_df['season'].drop_duplicates().to_list()
    except:
        seasons = 'no specified season(s)'

    print(f'Aggregating game-by-game stats dataframe containing season(s): {seasons}...')
    start = time.perf_counter()

    #Not all stats will have every column in the default group_by
    group_by = [col for col in group_by if col in games_df.columns]

    #If the stats provided are not by game (or don't include game_id for any reason) then it can be ignored
    try:
        games_df['game_id']
        
        #Add game date to games_df to (possibly) filter by
        schedule = pd.read_csv(schedule_path)[['game_id','game_date']]
        
        for df in [schedule, games_df]:
            df['game_id'] = df['game_id'].astype('Int64')

        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
        games_df = pd.merge(
            games_df.drop(columns=['game_date'], errors='ignore'), 
            schedule, 
            how='left'
        )
    except KeyError:
        pass

    #Apply pre-filter provided in params
    if params:
        games_df = apply_params(games_df, group_by, params)

    #Remove unwanted columns
    remove = [
        col for col in games_df.columns
        if (
                (
                    col in exclude
                    or any(s in col for s in NON_TOTALS)
                )
            and col not in group_by
            and col not in DEFAULT_AGG
            and col not in ['game_id', 'game_date', 'position']
            and col not in manual_agg.keys()
            and col != 'games_played'
        )
    ]
    games_df = games_df.drop(columns=remove, errors='ignore')

    #If game id is in the dataframe then it is a game-by-game stats dataframe
    #The game_id is stored as games_played
    if 'game_id' in games_df.columns:
        gbg = True
        games_df['games_played'] = games_df['game_id']
    else:
        gbg = False

    #Prepare aggregation clause
    clause = (
        {col: 'last' for col in BIO_STAT_COL if col in games_df.columns and col not in group_by and col != 'position'} | #Biographical info comes at the beginning of the dataframe (for nearly every single player every instance of bio info is the exact same)
        {col: concat_col_values for col in DEFAULT_AGG if col not in group_by and col in games_df.columns and col != 'position'}| #Columns that appear in the default group-by which are not included in the argument will concat everything pertaining to the group_by argument.
        ({'position': lambda x: x.mode().iloc[0]} if 'position' in games_df.columns else {})|
        ({'games_played': 'nunique' if gbg else 'max'} if 'games_played' in games_df.columns else {})|
        {col: partial(sum_unique_games, games_df_ref=games_df, col_name=col) for col in STANDINGS_COLS if col in games_df.columns and 'game_id' in games_df.columns}| #Standings columns must sum by game_id rather than summing every row (in case multiple rows are provided by another group)        {col: 'sum' for col in games_df.columns if not any(s in col for s in NON_TOTALS) and col not in group_by and col not in BIO_STAT_COL and col not in DEFAULT_AGG and col not in exclude} | #Base stats to sum (and then compare later on)
        {col: 'sum' for col in STANDINGS_COLS if col in games_df.columns and 'game_id' not in games_df.columns}| #Standings col summation when not summing by game_id
        {col: 'sum' for col in games_df.columns if not any(s in col for s in NON_TOTALS) and col not in group_by and col not in BIO_STAT_COL and col not in DEFAULT_AGG and col and col not in STANDINGS_COLS and col not in exclude} | #Base stats to sum (and then compare later on)
        {col: 'max' for col in games_df.columns if 'max_' in col or 'top_' in col} | #If stats are provided with certain labels they will be processed in their own manner
        {col: 'min' for col in games_df.columns if 'min_' in col or 'bottom_' in col} |
        {col: 'mean' for col in games_df.columns if 'avg_' in col or 'mean_' in col} |
        {col: 'median' for col in games_df.columns if 'count_' in col} |
        {col: 'std' for col in games_df.columns if 'std_' in col} |
        {col: 'var' for col in games_df.columns if 'var_' in col or 'variance_' in col} |
        {col: 'size' for col in games_df.columns if 'size_' in col}
    )

    #manual_agg is added after the initial prep in order to update values that may originally exist
    clause.update(manual_agg)

    #If groupby list is blank, sum everything
    if not group_by:
        games_df['_all'] = '_all'
        group_by = ['_all']

    #Apply group-by
    complete = games_df.groupby(by=group_by,as_index=False).agg(clause)

    #Apply post-filter provided in params
    if params:
        complete = apply_params(complete, group_by, params, 'after')
    
    #Add percentage stats if possible
    complete = extra_calc(complete, metrics=metrics)
    
    #Add roster information (this comes before to ensure bio columns, such as position, can be grouped-by if desired)
    if roster_path:
        complete = apply_rosters(complete, 'team' if 'player_id' not in group_by else 'player', roster_path)

    #Add per 60 stats and percentile rank
    complete = rank_stats(complete, rates, comparison, group_by)
    
    #Use default sorting if none is provided
    if not sort:
        sort = {col: True for col in ['player_name', 'season', 'team_abbr', 'player_id'] if col in complete.columns}

    sort_by = list(sort.keys())
    sort_asc = list(sort.values())
    
    complete = complete[[col for col in FRONT_COL if col in complete.columns]+[col for col in complete.columns if col not in FRONT_COL]]
    complete = complete.sort_values(by=sort_by, ascending=sort_asc) if sort_by else complete

    end = time.perf_counter()
    length = end-start
    elapsed = length if length < 60 else length / 60
    elapsed_unit = 'seconds' if length < 60 else 'minutes'
    print(f'...finished in {elapsed:.2f} {elapsed_unit}.')

    return complete

def nhl_plot_events(
        pbp:pd.DataFrame,
        group:Literal['skater','goalie','team','coach','game'],
        entities:int | str | list[int] | list[str],
        events:Union[Literal['all'], str, list[str]] = FENWICK_EVENTS,
        season:int | list[int] | None = None,
        strengths:Union[Literal['all'], str, list[str]] = 'all',
        season_types: int | list[int] = 2,
        strengths_title:str | None = None,
        marker_dict:dict = EVENT_MARKERS,
        team_colors:dict = {'away':'primary','home':'primary'},
        titles:str| list[str] | None = None,
        legend:bool = False,
        rotation:int | None = 0,
        display_range:str = 'full'
    ):
    """
    Given play-by-play data, plot arbitrary event locations for a group of entities.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data.
        group (Literal['skater','goalie','team','coach','game']):
            Entity type to plot (skater, goalie, team, coach, or game).
        entities (int|str|list[int]|list[str]):
            List of entities for the specified `group`:
            - skater/goalie: NHL API player_id(s)
            - team: team_abbr(s) (e.g. 'BOS')
            - coach: coach name(s) as stored in `pbp`
            - game: game_id(s)
        events (str or list[str] or 'all', optional):
            Event types to plot. Defaults to `wsba.FENWICK_EVENTS`. Use 'all' to plot all `wsba.EVENTS`.
        season (int|list[int]|None):
            If provided, filters season(s). If an int is provided with multiple `entities`, that season is used for all.
            If a list is provided, it must align one-to-one with `entities`. If None, seasons are inferred from `pbp`.
        strengths (str or list[str] or 'all', optional):
            Strength states to include. Default is 'all'.
        season_types (int or list[int], optional):
            Season type(s) to include. Default is 2 (regular season).
        strengths_title (str or None, optional):
            Optional label for the selected strengths (used on non-game plots).
        marker_dict (dict, optional):
            Mapping from event_type to matplotlib marker.
        team_colors (dict, optional):
            For game plots, selects 'primary' or 'secondary' for away/home team colors.
        titles (str or list[str] or None, optional):
            Optional title(s) aligned with `entities`.
        legend (bool, optional):
            If True, show a legend.
        display_range (str, optional):
            Rink display range. Passed to `wsba_rink()` / `hockey_rink.NHLRink.draw()` (e.g. 'full', 'offense', 'defense').
            Default is 'full'.
        rotation (int or None, optional):
            Rink rotation (degrees). Default is 0.

    Returns:
        dict:
            A dictionary of matplotlib figures: {entity: fig}.
    """

    if entities is None:
        entities = []
    elif not isinstance(entities, list):
        if isinstance(entities, (str, Integral)):
            entities = [entities]
        else:
            try:
                entities = list(entities)
            except TypeError:
                entities = [entities]

    if events == 'all':
        events = EVENTS
    elif isinstance(events, str):
        events = [events]

    if isinstance(season_types, int):
        season_types = [season_types]

    if isinstance(strengths, str) and strengths != 'all':
        strengths = [strengths]

    if isinstance(titles, str):
        titles = [titles] * len(entities)
    elif not titles:
        titles = []
    while len(entities) > len(titles):
        titles.append(None)

    if season is None:
        seasons = [None] * len(entities)
    elif isinstance(season, int):
        seasons = [season] * len(entities)
    else:
        seasons = season
        if len(seasons) != len(entities):
            raise ValueError("If `season` is a list, it must be the same length as `entities`.")

    pbp0 = pbp
    if season_types:
        pbp0 = pbp0.loc[pbp0['season_type'].isin(season_types)]

    if strengths != 'all':
        pbp0 = pbp0.loc[(pbp0['strength_state'].isin(strengths)) | (pbp0['strength_state'].astype(str).str[::-1].isin(strengths))]

    pbp_plot = pbp0.loc[pbp0['event_type'].isin(events)] if events else pbp0

    roster = pd.read_csv(DEFAULT_ROSTER)
    team_data = load_teaminfo()
    primary_color_by_wsba_id = team_primary_color_map(team_data)

    results = {}

    for title, entity, target_season in zip(titles, entities, seasons):
        pbp_season = pbp0 if target_season is None else pbp0.loc[pbp0['season'] == target_season]
        pbp_season_plot = pbp_plot if target_season is None else pbp_plot.loc[pbp_plot['season'] == target_season]

        if group == 'game':
            game_id = int(entity)
            game_rows = pbp_season_plot.loc[pbp_season_plot['game_id'] == game_id]
            if game_rows.empty:
                continue

            away_abbr = game_rows['away_team_abbr'].iloc[0]
            home_abbr = game_rows['home_team_abbr'].iloc[0]
            date = game_rows['game_date'].iloc[0]
            season_val = int(game_rows['season'].iloc[0])

            away_rows = team_data.loc[team_data['wsba_id'] == f'{away_abbr}{season_val}']
            home_rows = team_data.loc[team_data['wsba_id'] == f'{home_abbr}{season_val}']

            away_row = away_rows.iloc[0] if not away_rows.empty else None
            home_row = home_rows.iloc[0] if not home_rows.empty else None

            away_color_type = team_colors['away']
            home_color_type = team_colors['home']
            away_color_key = f"{away_color_type}_color"
            home_color_key = f"{home_color_type}_color"
            away_color_raw = away_row[away_color_key] if away_row is not None and away_color_key in away_row else '#1f77b4'
            away_color = (
                '#000000' if away_row is not None and away_color_type == 'secondary' and away_row.get('secondary_color') == '#FFFFFF' else away_color_raw
            )
            home_color = home_row[home_color_key] if home_row is not None and home_color_key in home_row else '#d62728'

            game_rows = game_rows.copy()
            game_rows['color'] = np.where(game_rows['event_team_abbr'] == away_abbr, away_color, home_color)

            plot_title = title if title is not None else f'{away_abbr} @ {home_abbr} - {date}'
            results[game_id] = plot_events(
                game_rows,
                events,
                title=plot_title,
                marker_dict=marker_dict,
                legend=legend,
                display_range=display_range,
                rotation=rotation,
            )
            continue

        if group == 'team':
            team_abbr = str(entity).upper()
            rows = pbp_season_plot.loc[pbp_season_plot['event_team_abbr'].astype(str).str.upper() == team_abbr]
            if rows.empty:
                continue

            rows = rows.copy()
            rows = apply_primary_colors(rows, primary_color_by_wsba_id)

            plot_title = title if title is not None else f'{team_abbr} Events'
            results[team_abbr] = plot_events(
                rows,
                events,
                title=plot_title,
                marker_dict=marker_dict,
                legend=legend,
                rotation=rotation,
                display_range=display_range,
            )
            continue

        if group == 'skater':
            player_id = int(entity)
            player_name = roster.loc[roster['player_id'] == player_id, 'player_name']
            player_name = player_name.iloc[0].title() if not player_name.empty else str(player_id)
            rows = pbp_season_plot.loc[pbp_season_plot['event_player_1_id'] == player_id]
            if rows.empty:
                continue

            rows = rows.copy()
            rows = apply_primary_colors(rows, primary_color_by_wsba_id)

            plot_title = title if title is not None else f'{player_name} Events'
            results[player_id] = plot_events(
                rows,
                events,
                title=plot_title,
                marker_dict=marker_dict,
                legend=legend,
                rotation=rotation,
                display_range=display_range,
            )
            continue

        if group == 'goalie':
            goalie_id = int(entity)
            goalie_name = roster.loc[roster['player_id'] == goalie_id, 'player_name']
            goalie_name = goalie_name.iloc[0].title() if not goalie_name.empty else str(goalie_id)
            if 'event_goalie_id' not in pbp_season_plot.columns:
                continue
            rows = pbp_season_plot.loc[pbp_season_plot['event_goalie_id'] == goalie_id]
            if rows.empty:
                continue

            rows = rows.copy()
            rows = apply_primary_colors(rows, primary_color_by_wsba_id)

            plot_title = title if title is not None else f'{goalie_name} Goalie Events'
            results[goalie_id] = plot_events(
                rows,
                events,
                title=plot_title,
                marker_dict=marker_dict,
                legend=legend,
                rotation=rotation,
                display_range=display_range,
            )
            continue

        if group == 'coach':
            coach = str(entity)
            if 'home_coach' not in pbp_season_plot.columns or 'away_coach' not in pbp_season_plot.columns:
                continue
            coached = pbp_season_plot.loc[
                ((pbp_season_plot['home_coach'] == coach) & (pbp_season_plot['event_team_abbr'] == pbp_season_plot['home_team_abbr']))
                | ((pbp_season_plot['away_coach'] == coach) & (pbp_season_plot['event_team_abbr'] == pbp_season_plot['away_team_abbr']))
            ]
            if coached.empty:
                continue

            coached = coached.copy()
            coached = apply_primary_colors(coached, primary_color_by_wsba_id)

            plot_title = title if title is not None else f'{coach} Events'
            results[coach] = plot_events(
                coached,
                events,
                title=plot_title,
                marker_dict=marker_dict,
                legend=legend,
                rotation=rotation,
                display_range=display_range,
            )
            continue

        raise ValueError(f"Unknown group: {group}")

    if strengths_title and group != 'game':
        # Add strengths text to all non-game figures (keeps output similar to other plotting helpers).
        for fig in results.values():
            fig.text(0.5, 0.07, 'Strength(s)', ha='center', fontsize=10)
            fig.text(0.5, 0.03, f'{strengths_title}', ha='center', fontsize=10)

    return results

def repo_load_rosters(seasons: int | list[int] | None = None) -> pd.DataFrame:
    """
    Returns roster data from repository

    Args:
        seasons (int | list[int] | None, optional):
            Season or seasons to return. If None, all repository roster data is returned.

    Returns:
        pd.DataFrame:
            A DataFrame containing roster data for supplied seasons.
    """

    if isinstance(seasons, int):
        seasons = [seasons]
        
    data = pd.read_csv(DEFAULT_ROSTER)
    if seasons:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_schedule(seasons: int | list[int] | None = None) -> pd.DataFrame:
    """
    Returns schedule data from repository

    Args:
        seasons (int | list[int] | None, optional):
            Season or seasons to return. If None, all repository schedule data is returned.

    Returns:
        pd.DataFrame:
            A DataFrame containing the schedule data for the specified season and date range.    
    """

    if isinstance(seasons, int):
        seasons = [seasons]

    data = pd.read_csv(SCHEDULE_PATH, low_memory=False)
    if seasons:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_teaminfo() -> pd.DataFrame:
    """
    Returns team data from repository

    Args:

    Returns:
        pd.DataFrame:
            A DataFrame containing general team information.
    """

    return pd.read_csv(INFO_PATH)

def utility_get_schema(df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns schema for provided dataframe

    Args:
        df (pd.DataFrame):
            Any dataframe generated by functions in the wsba-hockey package

    Returns:
        pd.DataFrame:
            A DataFrame containing the schema for the specified dataframe.    
    """

    return pd.DataFrame({
                'column': df.columns,
                'dtype': df.dtypes
            })

def utility_get_unique(df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns unique values in each column for provided dataframe.

    Args:
        df (pd.DataFrame):
            Any dataframe generated by functions in the wsba-hockey package

    Returns:
        pd.DataFrame:
            A DataFrame containing the unique values in each column for the specified dataframe.    
    """

    unique_dict = {
        col: pd.Series(df[col].dropna().unique())
        for col in df.columns
    }

    unique_df = pd.DataFrame(dict(
        [(k, pd.Series(v)) for k, v in unique_dict.items()]
    ))

    return unique_df
