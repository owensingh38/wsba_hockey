import re
import warnings
import os
import time
import numpy as np
import pandas as pd
import requests as rs
import json as json_lib
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from wsba_hockey.tools.globals import (
    COL_MAP,
    DEFAULT_ROSTER,
    EDGE_CAT,
    EVENTS,
    FENWICK_EVENTS,
    PBP_COLS,
    POS_BASE_PROB,
    SHOT_TYPES,
    STRENGTHS,
    convert_to_seconds,
    get_contents,
    get_soup,
    get_team,
)
warnings.filterwarnings('ignore')

### SCRAPING FUNCTIONS ###
# Provided in this file are functions vital to the scraping functions in the WSBA Hockey Python package. #

## ORDER OF OPERATIONS ##
# Create game information to use with all functions
# Retreive JSON data
# Parse JSON data
# Retreive and clean HTML pbp with player information
# Parse HTML pbp, return parsed HTML
# Combine pbp data
# Retreive and analyze HTML shifts with player information for home and away teams
# Parse shift events
# Combine all data, return complete play-by-play

def adjust_coords(pbp):
    #Given JSON or ESPN pbp data, return pbp with adjusted coordinates

    #Recalibrate coordinates
    #Determine the direction teams are shooting in a given period
    pbp['med_x'] = (pbp.where(pbp['event_type'].isin(['missed-shot','shot-on-goal','goal'])).groupby(['event_team_venue','period','game_id'])['x'].transform('median'))

    pbp = pbp.reset_index(drop=True)

    #Adjust coordinates
    pbp['x_adj'] = np.where((((pbp['event_team_venue']=='home')&(pbp['med_x'] < 0))|((pbp['event_team_venue']=='away')&(pbp['med_x'] > 0))),-pbp['x'],pbp['x'])

    #Adjust y if necessary
    pbp['y_adj'] = np.where((pbp['x']==pbp['x_adj']),pbp['y'],-pbp['y'])

    #Calculate event distance and angle relative to venue location
    pbp['event_distance'] = np.where(pbp['event_team_venue']=='home',np.sqrt(((89 - pbp['x_adj'])**2) + (pbp['y_adj']**2)),np.sqrt((((-89) - pbp['x_adj'])**2) + (pbp['y_adj']**2)))
    pbp['event_angle'] = np.where(pbp['event_team_venue']=='home',np.degrees(np.arctan2(abs(pbp['y_adj']), abs(89 - pbp['x_adj']))),np.degrees(np.arctan2(abs(pbp['y_adj']), abs((-89) - pbp['x_adj']))))

    #Adjusted coordinates move away shots to the left side of the ice and home shots to the right side of the ice
    #Fixed shots are mapped to the same side of the ice (the right side)
    pbp['x_fixed'] = np.abs(pbp['x_adj'])
    pbp['y_fixed'] = np.where(pbp['x_adj']<0,pbp['y_adj']*-1,pbp['y_adj'])

    #Return: pbp with adjiusted coordinates
    return pbp


def fix_players(pbp):
    # Add/fix player info for shooters and goaltenders.
    try:
        find = pbp.loc[
            (pbp['event_type'].isin(FENWICK_EVENTS)) & (pbp['event_player_1_hand'].isna()),
            'event_player_1_id'
        ].drop_duplicates().to_list()
    except Exception:
        find = pbp['event_player_1_id'].drop_duplicates().to_list()

    roster = pd.read_csv(DEFAULT_ROSTER)
    roster = roster.loc[
        roster['player_id'].isin(find)
    ].drop_duplicates(['player_id'])[['player_name', 'player_id', 'position', 'handedness']]

    if find:
        print('Adding player info to pbp...')

        missing = pbp.loc[
            (
                (pbp['event_player_1_id'].isin(find)) &
                (~pbp['event_player_1_id'].isin(roster['player_id']))
            ),
            'event_player_1_id'
        ].drop_duplicates().dropna().to_list()
        if missing:
            from wsba_hockey.wsba_main import nhl_scrape_player_info

            add = nhl_scrape_player_info(missing)[['player_name', 'player_id', 'handedness']]
            roster = pd.concat([roster, add]).reset_index(drop=True)

        roster['player_id'] = roster['player_id'].astype('Int64')
        hand_dict = roster.set_index('player_id').to_dict()['handedness']

        pbp['event_goalie_id'] = np.where(
            pbp['event_team_venue'] == 'away',
            pbp['home_goalie_id'],
            pbp['away_goalie_id']
        )
        pbp['event_player_1_hand'] = (
            pbp['event_player_1_id'].astype('Int64').map(hand_dict).where(lambda x: x.notna(), None)
        )

    return pbp


def apply_passing_imputation(pbp):
    # Estimate player passing/setting impacts on shot attempts.
    goals = pbp['event_type'] == 'goal'
    non_goals = pbp['event_type'].isin(FENWICK_EVENTS) & (pbp['event_type'] != 'goal')

    for venue in ['away', 'home']:
        team_mask = pbp['event_team_venue'] == venue
        player_cols = [f'{venue}_on_{j}_id' for j in range(1, 7)]
        pos_cols = [f'{venue}_on_{j}_pos' for j in range(1, 7)]
        prob_cols = {
            'primary': [f'{venue}_on_{j}_primary_fenwick_assist_probability' for j in range(1, 7)],
            'secondary': [f'{venue}_on_{j}_secondary_fenwick_assist_probability' for j in range(1, 7)],
            'tertiary': [f'{venue}_on_{j}_tertiary_fenwick_assist_probability' for j in range(1, 7)]
        }

        for assist_type in prob_cols:
            pbp[prob_cols[assist_type]] = 0.0

        goal_team_mask = team_mask & goals
        for j in range(1, 7):
            player_col = f'{venue}_on_{j}_id'
            is_primary_assist = pbp[player_col] == pbp['event_player_2_id']
            is_secondary_assist = pbp[player_col] == pbp['event_player_3_id']

            pbp.loc[goal_team_mask & is_primary_assist, f'{venue}_on_{j}_primary_fenwick_assist_probability'] = 1.0
            pbp.loc[goal_team_mask & is_secondary_assist, f'{venue}_on_{j}_secondary_fenwick_assist_probability'] = 1.0

        if goal_team_mask.any():
            goal_indices = pbp.index[goal_team_mask]
            goal_on_ice_ids = pbp.loc[goal_indices, player_cols].values
            goal_on_ice_pos = pbp.loc[goal_indices, pos_cols].values
            tertiary_probs_goals = np.vectorize(
                lambda pos: POS_BASE_PROB['tertiary'].get(pos, 0)
            )(goal_on_ice_pos)

            goal_player_1 = pbp.loc[goal_indices, 'event_player_1_id'].values[:, None]
            goal_player_2 = pbp.loc[goal_indices, 'event_player_2_id'].values[:, None]
            goal_player_3 = pbp.loc[goal_indices, 'event_player_3_id'].values[:, None]
            involved_mask = (
                (goal_on_ice_ids == goal_player_1) |
                (goal_on_ice_ids == goal_player_2) |
                (goal_on_ice_ids == goal_player_3)
            )
            tertiary_probs_goals[involved_mask] = 0
            tertiary_sums = tertiary_probs_goals.sum(axis=1, keepdims=True)
            tertiary_sums[tertiary_sums == 0] = 1
            tertiary_probs_goals = (tertiary_probs_goals / tertiary_sums) * 0.8
            pbp.loc[goal_indices, prob_cols['tertiary']] = tertiary_probs_goals

        if non_goals.any():
            non_goal_team_mask = team_mask & non_goals
            non_goal_indices = pbp.index[non_goal_team_mask]
            on_ice_ids = pbp.loc[non_goal_indices, player_cols].values
            on_ice_pos = pbp.loc[non_goal_indices, pos_cols].values
            shooter_ids = pbp.loc[non_goal_indices, 'event_player_1_id'].values[:, None]

            probs = {}
            for assist_type in ['primary', 'secondary', 'tertiary']:
                probs[assist_type] = np.vectorize(
                    lambda pos: POS_BASE_PROB[assist_type].get(pos, 0)
                )(on_ice_pos)
                probs[assist_type][on_ice_ids == shooter_ids] = 0
                sums = probs[assist_type].sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                probs[assist_type] = (probs[assist_type] / sums) * 0.8

            for assist_type in ['primary', 'secondary', 'tertiary']:
                pbp.loc[non_goal_indices, prob_cols[assist_type]] = probs[assist_type]

    return pbp

## JSON FUNCTIONS ##
def get_game_roster(json):
    #Given raw json data, return game rosters
    roster = pd.json_normalize(json['rosterSpots'])
    roster['player_name'] = (roster['firstName.default'] + " " + roster['lastName.default']).str.upper()

    #Return: roster information
    return roster

def get_game_coaches(game_id):
    #Given game info, return head coaches for away and home team
    
    #Retreive data (or try to)
    try:
        json = rs.get(f'https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail').json()
        data = json['gameInfo']

        #Add coaches
        try:
            away = data['awayTeam']['headCoach']['default'].upper()
            home = data['homeTeam']['headCoach']['default'].upper()
            
            coaches = {'away':away,
                    'home':home}
        except KeyError:
            return {}

        #Return: dict with coaches
        return coaches
    except rs.JSONDecodeError:
        #Right-rail content is missing for some playoff games in 2019-20
        return {}
    
def get_game_info(game_id):
    #Given game_id, return game information
    
    def _cached_get_json(session: rs.Session, url: str, cache_path: str, max_age_sec: int = 120):
        try:
            if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < max_age_sec):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json_lib.load(f)
        except Exception:
            pass

        data = session.get(url).json()
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json_lib.dump(data, f)
        except Exception:
            pass
        return data

    #Retrieve data (parallelize independent endpoints to reduce wall time)
    session = rs.Session()
    api_pbp = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    api_shifts = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    api_right_rail = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"

    cache_root = os.environ.get("WSBA_HOCKEY_CACHE_DIR", ".wsba_cache")
    pbp_cache = os.path.join(cache_root, "api", "pbp", f"{game_id}.json")
    shifts_cache = os.path.join(cache_root, "api", "shiftcharts", f"{game_id}.json")
    rr_cache = os.path.join(cache_root, "api", "right_rail", f"{game_id}.json")

    with ThreadPoolExecutor(max_workers=3) as executor:
        pbp_future = executor.submit(_cached_get_json, session, api_pbp, pbp_cache)
        shifts_future = executor.submit(_cached_get_json, session, api_shifts, shifts_cache)
        rr_future = executor.submit(_cached_get_json, session, api_right_rail, rr_cache)

        pbp_json = pbp_future.result()
        shifts_json = shifts_future.result()

        # Right-rail content is missing for some games; preserve prior behavior (empty dict)
        coaches = {}
        try:
            rr_json = rr_future.result()
            game_info = rr_json.get('gameInfo', {})
            away = game_info.get('awayTeam', {}).get('headCoach', {}).get('default')
            home = game_info.get('homeTeam', {}).get('headCoach', {}).get('default')
            if away and home:
                coaches = {'away': away.upper(), 'home': home.upper()}
        except Exception:
            coaches = {}

    #Provide explicit error for games which have not yet occured
    if pbp_json['gameState'] in ['FUT', 'PRE']:
        raise ValueError("Game has not yet occured.")

    #Games don't always have JSON shifts, for whatever reason
    json_shifts = pd.json_normalize(shifts_json.get('data', []))
    if shifts_json.get('total', 0) == 0:
        json_shifts = pd.DataFrame()

    #Split information
    base = pd.json_normalize(pbp_json)
    game_id = base['id'][0]
    season = base['season'][0]
    season_type = base['gameType'][0]
    game_date = base['gameDate'][0]
    game_state = base['gameState'][0]
    start_time = base['startTimeUTC'][0]
    venue = base['venue.default'][0]
    venue_location = base['venueLocation.default'][0]
    away_team_id = base['awayTeam.id'][0]
    away_team_abbr = base['awayTeam.abbrev'][0]
    home_team_id = base['homeTeam.id'][0]
    home_team_abbr = base['homeTeam.abbrev'][0]

    #Add roster
    roster = get_game_roster(pbp_json)
    #In the HTML parsing process, player are identified by a regex pattern (ABB #00 such as BOS #37) or number and name in the following format: #00 NAME (i.e. #37 BERGERON) so these are added as IDs of sorts.  
    roster['descID'] = '#'+roster['sweaterNumber'].astype(str)+" "+roster['lastName.default'].str.upper()
    roster['team_abbr'] = roster['teamId'].replace({
        away_team_id:[away_team_abbr],
        home_team_id:[home_team_abbr]
    })
    roster['key'] = roster['team_abbr'] + " #" + roster['sweaterNumber'].astype(str)

    #Create an additional roster dictionary for use with HTML parsing
    roster_dict = {'away':{}, 'home':{}}
    
    #Evaluate and add players by team
    for team in ['away','home']:
        abbr = (away_team_abbr if team == 'away' else home_team_abbr)
        rost = roster.loc[roster['team_abbr']==abbr]
        
        #Now iterate through team players
        for player,id,num,pos,team_abbr,key in zip(rost['player_name'],rost['playerId'],rost['sweaterNumber'],rost['positionCode'],rost['team_abbr'],rost['key']):
            roster_dict[team].update({str(num):[key, pos, player, team_abbr, id]})

    #Return: game information
    return {"game_id":str(game_id),
            "season":season,
            "season_type":season_type,
            "game_date":game_date,
            "game_state":game_state,
            "start_time":start_time,
            'venue':venue,
            'venue_location':venue_location,
            'away_team_id':away_team_id,
            'away_team_abbr':away_team_abbr,
            'home_team_id':home_team_id,
            'home_team_abbr':home_team_abbr,
            'events':pd.json_normalize(pbp_json['plays']).reset_index(drop=True),
            'rosters':roster,
            'HTML_rosters':roster_dict,
            'coaches':coaches,
            'json_shifts':json_shifts}

def parse_json(info):
    #Given game info, return JSON document

    #Retreive data
    events = info['events']

    #Return error if game is set in the future
    if info['game_state'] == 'FUT':
        raise ValueError(f"Game {info['id'][0]} has not occured yet.")
    
    #Test columns
    cols = ['eventId', 'timeInPeriod', 'timeRemaining', 'situationCode', 'homeTeamDefendingSide', 'typeCode', 'typeDescKey', 'sortOrder', 'periodDescriptor.number', 'periodDescriptor.periodType', 'periodDescriptor.maxRegulationPeriods', 'details.eventOwnerTeamId', 'details.losingPlayerId', 'details.winningPlayerId', 'details.xCoord', 'details.yCoord', 'details.zoneCode', 'pptReplayUrl', 'details.shotType', 'details.scoringPlayerId', 'details.scoringPlayerTotal', 'details.assist1PlayerId', 'details.assist1PlayerTotal', 'details.assist2PlayerId', 'details.assist2PlayerTotal', 'details.goalieInNetId', 'details.awayScore', 'details.homeScore', 'details.highlightClipSharingUrl', 'details.highlightClipSharingUrlFr', 'details.highlightClip', 'details.highlightClipFr', 'details.discreteClip', 'details.discreteClipFr', 'details.shootingPlayerId', 'details.awaySOG', 'details.homeSOG', 'details.playerId', 'details.hittingPlayerId', 'details.hitteePlayerId', 'details.reason', 'details.typeCode', 'details.descKey', 'details.duration', 'details.servedByPlayerId', 'details.secondaryReason', 'details.blockingPlayerId', 'details.committedByPlayerId', 'details.drawnByPlayerId', 'game_id', 'season', 'season_type', 'game_date']

    for col in cols:
        try:events[col]
        except:
            events[col]=""

    #Event_player_columns include players in a given set of events; the higher the number, the greater the importance the event player was to the play
    events['event_player_1_id'] = events['details.winningPlayerId'].combine_first(events['details.scoringPlayerId'])\
                                                                   .combine_first(events['details.shootingPlayerId'])\
                                                                   .combine_first(events['details.playerId'])\
                                                                   .combine_first(events['details.hittingPlayerId'])\
                                                                   .combine_first(events['details.committedByPlayerId'])
        
    events['event_player_2_id'] = events['details.losingPlayerId'].combine_first(events['details.assist1PlayerId'])\
                                                                    .combine_first(events['details.hitteePlayerId'])\
                                                                    .combine_first(events['details.drawnByPlayerId'])\
                                                                    .combine_first(events['details.blockingPlayerId'])

    events['event_player_3_id'] = events['details.assist2PlayerId']

    events['event_team_venue'] = np.where(events['details.eventOwnerTeamId']==info['home_team_id'],"home","away")

    events['event_team_abbr'] = events['details.eventOwnerTeamId'].replace({
        info['away_team_id']:[info['away_team_abbr']],
        info['home_team_id']:[info['home_team_abbr']]
    })

    #Rename columns to follow WSBA naming conventions
    events = events.rename(columns={
        "eventId":"event_id",
        "periodDescriptor.number":"period",
        "periodDescriptor.periodType":"period_type",
        "timeInPeriod":"period_time_elasped",
        "timeRemaining":"period_time_remaining",
        "situationCode":"situation_code",
        "homeTeamDefendingSide":"home_team_defending_side",
        "typeCode":"event_type_code",
        "typeDescKey":"event_type",
        "pptReplayUrl":"ppt_replay_url",
        "details.shotType":"shot_type",
        "details.duration":"penalty_duration",
        "details.descKey":"penalty_type",
        "details.typeCode":'penalty_attribution',
        "details.reason":"event_reason",
        "details.zoneCode":"zone_code",
        "details.xCoord":"x",
        "details.yCoord":"y",
        "details.goalieInNetId": "event_goalie_id",
        "details.awaySOG":"away_sog",
        "details.homeSOG":"home_sog"
    })

    #Coordinate adjustments:
    # x, y - Raw coordinates from JSON pbp
    # x_adj, y_adj - Adjusted coordinates configuring the away offensive zone to the left and the home offensive zone to the right
    #Some games (mostly preseason and all star games) do not include coordinates. 
    if info['season'] in [20052006, 20062007, 20072008, 20082009, 20092010]:
        #If the json is used as a supplement for the ESPN pbp data then remove unnecessary columns
        events = events.drop(columns=['x','y','event_team_venue','period_seconds_elapsed','game_id',
                                      'period_time_elapsed', 'shot_type', 'zone_code', 'event_player_1_id', 'event_player_2_id', 'event_player_3_id'],
                                      errors='ignore')
    else:
        try:
            events = adjust_coords(events)
        except KeyError:
            print(f"No coordinates found for game {info['game_id'][0]}...")
            events['x_adj'] = np.nan
            events['y_adj'] = np.nan
            events['event_distance'] = np.nan
            events['event_angle'] = np.nan
        
    #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
    events['period_seconds_elapsed'] = events['period_time_elasped'].apply(convert_to_seconds)
    events['seconds_elapsed'] = ((events['period']-1)*1200)+events['period_seconds_elapsed']

    events = events.loc[(events['event_type']!="")]

    #Return: dataframe with parsed game
    return events

## HTML PBP FUNCTIONS ##
def strip_html_pbp(td,rosters):
    #Given html row, parse data from HTML pbp
    #Harry Shomer's Code (modified)
    
    #HTML Parsing
    for y in range(len(td)):
        # Get the 'br' tag for the time column...this get's us time remaining instead of elapsed and remaining combined
        if y == 3:
            td[y] = td[y].get_text()   # This gets us elapsed and remaining combined-< 3:0017:00
            index = td[y].find(':')
            td[y] = td[y][:index+3]
        elif (y == 6 or y == 7) and td[0] != '#':
            # 6 & 7-> These are the player 1 ice one's
            # The second statement controls for when it's just a header
            baz = td[y].find_all('td')
            bar = [baz[z] for z in range(len(baz)) if z % 4 != 0]  # Because of previous step we get repeats...delete some

            # The setup in the list is now: Name/Number->Position->Blank...and repeat
            # Now strip all the html
            players = []
            for i in range(len(bar)):
                if i % 3 == 0:
                    try:
                        #Using the supplied json we can bind player name and id to number and team
                        #Find number and team of player then lookup roster dictionary
                        
                        number = bar[i].get_text().strip('\n')  # Get number and strip leading/trailing newlines
                        if y == 6:
                            team = 'away'
                        else:
                            team = 'home'
                        
                        id = rosters[team][str(number)][4]
                        name = rosters[team][str(number)][2]
                        position = rosters[team][str(number)][1]
                        
                    except KeyError:
                        name = ''
                        number = ''
                        id = ''
                elif i % 3 == 1:
                    if name != '':
                        players.append([name, number, position, id])

            td[y] = players
        else:
            td[y] = td[y].get_text()

    return td


def clean_html_pbp(info):
    #Harry Shomer's Code (modified)

    game_id = info['game_id']
    #Retreive data
    season = info['season']
    doc = f"https://www.nhl.com/scores/htmlreports/{season}/PL{game_id[-6:]}.HTM"

    # Transparent cache to avoid repeated downloads (useful when iterating on the same game).
    cache_root = os.environ.get("WSBA_HOCKEY_CACHE_DIR", ".wsba_cache")
    cache_path = os.path.join(cache_root, "htmlreports", str(season), f"PL{game_id[-6:]}.HTM")
    profile = os.environ.get("WSBA_HOCKEY_PROFILE", "").lower() in {"1", "true", "yes"}
    t0 = time.perf_counter() if profile else None

    html = None
    try:
        dynamic = info.get("game_state") not in {"OFF", "FINAL"}
        if os.path.exists(cache_path) and (not dynamic or (time.time() - os.path.getmtime(cache_path) < 120)):
            with open(cache_path, "rb") as f:
                html = f.read()
    except Exception:
        html = None

    if html is None:
        html = rs.get(doc).content
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(html)
        except Exception:
            pass

    if profile and t0 is not None:
        print(f" [html pbp fetch {(time.perf_counter()-t0):.2f}s]", end="")
        t0 = time.perf_counter()

    #Rosters (used by both fast and fallback parsing)
    rosters = info['HTML_rosters']

    # Fast path: parse the outer play-by-play table with lxml (substantially faster than BeautifulSoup).
    # Fall back to BeautifulSoup if lxml isn't available or the table structure isn't detected.
    try:
        from lxml import html as lxml_html  # type: ignore

        root = lxml_html.fromstring(html)
        trs = root.xpath("//tr[td[contains(@class,'bborder')]]")
        rows = []

        def _map_on_ice(cell, team_key: str):
            nums = [t.strip() for t in cell.xpath('.//font/text()') if t.strip()]
            team_roster = rosters[team_key]
            players = []
            for num in nums:
                try:
                    entry = team_roster[str(num)]
                    # [key, pos, player_name, team_abbr, player_id]
                    players.append([entry[2], str(num), entry[1], entry[4]])
                except KeyError:
                    continue
            return players

        for tr in trs:
            tds = tr.xpath("./td[contains(@class,'bborder')]")
            if len(tds) < 8:
                continue

            out = []
            for idx, td in enumerate(tds[:6]):
                # Preserve trailing spaces in the description column to match legacy BeautifulSoup output.
                raw = td.text_content().replace("\r", "").replace("\n", "")
                txt = raw.strip() if idx != 5 else raw.lstrip()
                if idx == 3:
                    # Match prior behavior: keep only the elapsed time portion.
                    colon = txt.find(":")
                    txt = txt[: colon + 3] if colon != -1 else txt
                out.append(txt)

            # On-ice players
            out.append(_map_on_ice(tds[6], "away"))
            out.append(_map_on_ice(tds[7], "home"))
            rows.append(out)

        if rows:
            if profile and t0 is not None:
                print(f" [html pbp parse {(time.perf_counter()-t0):.2f}s]", end="")
            return rows
    except Exception:
        pass

    soup = get_contents(html)

    # Create a list of lists (each length 8)...corresponds to 8 columns in html pbp
    td = [soup[i:i + 8] for i in range(0, len(soup), 8)]

    cleaned_html = [strip_html_pbp(x,rosters) for x in td]

    if profile and t0 is not None:
        print(f" [html pbp parse {(time.perf_counter()-t0):.2f}s]", end="")

    return cleaned_html

def parse_html(info):
    #Given game info, return HTML event data

    #Retreive game information and html events
    rosters = info['HTML_rosters']
    events = clean_html_pbp(info)

    teams = {info['away_team_abbr']:['away'],
             info['home_team_abbr']:['home']}
    
    #Parsing
    # Build a list of dicts and create a single DataFrame at the end (much faster than concatenating many 1-row frames).
    event_rows = []

    regex_team_num = re.compile(r'([A-Z]{2,3}|\b[A-Z]\.[A-Z])\s+#(\d+)')
    regex_goal = re.compile(r'#(\d+)\s+')
    regex_hash_num = re.compile(r'#\d+')
    for event in events:
        events_dict = {}
        if event[0] == "#" or event[4] in ['GOFF', 'EGT', 'PGSTR', 'PGEND', 'ANTHEM', 'SPC', 'PBOX', 'EISTR', 'EIEND','EGPID'] or event[3]=='-16:0-':
            continue
        else:
            #Event info
            events_dict['event_num'] = int(event[0])
            events_dict['period'] = int(event[1])
            events_dict['strength'] = re.sub(u'\xa0'," ",event[2])
            events_dict['period_time_elapsed'] = event[3]
            events_dict['seconds_elapsed'] = convert_to_seconds(event[3]) + (1200*(int(event[1])-1))
            events_dict['event_type'] = event[4]

            desc = re.sub(u'\xa0'," ",event[5])
            events_dict['description'] = desc

            events_dict['shot_type'] = desc.split(",")[1].lower().strip(" ") if event[4] in ['BLOCK','MISS','SHOT','GOAL'] else ""
            zone = [x for x in desc.split(',') if 'Zone' in x]
            if not zone:
                events_dict['zone_code'] = None
            elif zone[0].find("Off") != -1:
                events_dict['zone_code'] = 'O'
            elif zone[0].find("Neu") != -1:
                events_dict['zone_code'] = 'N'
            elif zone[0].find("Def") != -1:
                events_dict['zone_code'] = 'D'

            #Convert team names for compatiblity
            replace = [('LAK',"L.A"),('NJD',"N.J"),('SJS',"S.J"),('TBL',"T.B")]
            for name, repl in replace:
                desc = desc.replace(repl,name)
            
            event_team = desc[0:3] if desc[0:3] in teams.keys() else ""
            events_dict['event_team_abbr'] = event_team

            events_dict['away_team_abbr'] = info['away_team_abbr']
            events_dict['home_team_abbr'] = info['home_team_abbr']

            away_skaters = 0
            away_goalie = 0
            #Away on-ice
            for i in range(len(event[6])):
                player = event[6][i][0]
                pos = event[6][i][2]
                id = event[6][i][3]
                
                if pos == 'G':
                    events_dict['away_goalie'] = player
                    events_dict['away_goalie_id'] = id
                    away_goalie += 1
                else:
                    events_dict[f'away_on_{i+1}'] = player
                    events_dict[f'away_on_{i+1}_id'] = id
                    away_skaters += 1

            home_skaters = 0
            home_goalie = 0
            #Home on-ice
            for i in range(len(event[7])):
                player = event[7][i][0]
                pos = event[7][i][2]    
                id = event[7][i][3]
                
                if pos == 'G':
                    events_dict['home_goalie'] = player
                    events_dict['home_goalie_id'] = id
                    home_goalie += 1
                else:
                    events_dict[f'home_on_{i+1}'] = player
                    events_dict[f'home_on_{i+1}_id'] = id
                    home_skaters += 1
            
            event_players = []
            #Determine parsing route based on event
            if event[4] in ['FAC','HIT','BLOCK','PENL']:
                #Regex to find team and player number involved (finds all for each event)
                #Code is modified from Harry Shomer in order to account for periods in a team abbreviation
                fac = regex_team_num.findall(desc)
                #Filter incorrectly parsed teams
                repl = []
                for team, num in fac:
                    if team in teams.keys():
                        repl.append((team,num))
                fac = repl

                #Find first event player
                ep1_num = ''
                for i in range(len(fac)):
                    team, num = fac[i]
                    if team == event_team:
                        ep1_num = num
                        event_players.append(fac[i])
                    else:
                        continue
                    
                #Find other players
                for i in range(len(fac)):
                    team, num = fac[i]
                    if num == ep1_num:
                        continue
                    else:
                        event_players.append(fac[i])
            elif event[4]=='GOAL':
                #Parse goal
                goal = regex_goal.findall(desc)
                
                #Add all involved players
                for point in goal:
                    #In this loop, point is a player number.  We can assign event_team to all players in a goal
                    event_players.append((event_team,str(point)))
            elif event[4]=='DELPEN':
                #Don't parse DELPEN events 
                #These events typically have no text but when they do it is often erroneous or otherwise problematic

                ""
            else:
                #Parse single or no player events
                fac = regex_hash_num.findall(desc)

                for i in range(len(fac)):
                    num = fac[i].replace("#","")
                    event_players.append((event_team,str(num)))

            for i in range(len(event_players)):
                #For each player, evaluate their event data, then retreive information from rosters
                team, num = event_players[i]
                
                status = teams[team]
                data = rosters[status[0]]

                #In rare instances the event player is not on the event team (i.e. "WSH TAKEAWAY - #71 CIRELLI, Off. Zone" when #71 CIRELLI is on TBL)
                try:
                    events_dict[f'event_player_{i+1}_name'] = data[str(num)][2]
                    events_dict[f'event_player_{i+1}_id'] = data[str(num)][4]
                    events_dict[f'event_player_{i+1}_pos'] = data[str(num)][1]
                except:
                    ''

            #Event skaters and strength-state information
            events_dict['away_skaters'] = away_skaters
            events_dict['home_skaters'] = home_skaters
            events_dict['away_goalie_in'] = away_goalie
            events_dict['home_goalie_in'] = home_goalie

            event_skaters = away_skaters if info['away_team_abbr'] == event_team else home_skaters
            event_skaters_against = away_skaters if info['home_team_abbr'] == event_team else home_skaters
            events_dict['strength_state'] = f'{event_skaters}v{event_skaters_against}'
            events_dict['event_skaters'] = np.where(event_team == info['home_team_abbr'],home_skaters,away_skaters)

        event_rows.append(events_dict)

    data = pd.DataFrame(event_rows)
    data['event_type'] = data['event_type'].replace({
        "PGSTR": "pre-game-start",
        "PGEND": "pre-game-end",
        'GSTR':"game-start",
        "ANTHEM":"anthem",
        "PSTR":"period-start",
        "FAC":"faceoff",
        "SHOT":"shot-on-goal",
        "BLOCK":"blocked-shot",
        "STOP":"stoppage",
        "MISS":"missed-shot",
        "HIT":"hit",
        "GOAL":"goal",
        "GIVE":"giveaway",
        "TAKE":"takeaway",
        "DELPEN":"delayed-penalty",
        "PENL":"penalty",
        "CHL":"challenge",
        "SOC":'shootout-complete',
        "PEND":"period-end",
        "GEND":"game-end"
    })
    
    #Return: parsed HTML pbp
    return data

### ESPN SCRAPING FUNCTIONS ###
def espn_game_id(date,away,home):
    #Given a date formatted as YYYY-MM-DD and teams, return game id from ESPN schedule
    date = date.replace("-","")

    #Retreive data
    api = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={date}"
    schedule = pd.json_normalize(rs.get(api).json()['events'])

    #Create team abbreviation columns
    schedule['away_team_abbr'] = schedule['shortName'].str[:3].str.strip(" ")
    schedule['home_team_abbr'] = schedule['shortName'].str[-3:].str.strip(" ")
    
    #Modify team abbreviations as necessary
    schedule = schedule.replace({
        "LA":"LAK",
        "NJ":"NJD",
        "SJ":"SJS",
        "TB":"TBL",
    })

    #Retreive game id
    game_id = schedule.loc[(schedule['away_team_abbr']==away)&
                           (schedule['home_team_abbr']==home),'id'].tolist()[0]

    #Return: ESPN game id
    return game_id

def parse_espn(date,away,home):
    #Given a date formatted as YYYY-MM-DD and teams, return game events from ESPN
    game_id = espn_game_id(date,away,home)
    
    #Hidden ESPN API endpoint (akin to the gamecenter/{game_id}/play-by-play NHL endpoint)
    url = f'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={game_id}'
    data = rs.get(url).json()
    teams = data['boxscore']['teams']

    #Retreive plays
    espn_events = pd.json_normalize(data['plays']).rename(columns={
        'period.number':'period',
        'clock.displayValue':'period_time_elapsed',
        'coordinate.x':'x',
        'coordinate.y':'y',
        'type.text':'event_type',
    })
    
    #Some games are missing plays on ESPN, for some reason
    if espn_events.empty:
        print(f"No coordinates found for game ...")
        return pd.DataFrame(columns=['period','seconds_elapsed','event_type','event_team_abbr'])
    else:
        #Retreive event team venue with team data (maintain the team abbreviation fill-in at the bottom)
        espn_events['event_team_venue'] = espn_events['team.id'].replace({
            teams[0]['team']['id']: teams[0]['homeAway'],
            teams[1]['team']['id']: teams[1]['homeAway']
        })

        #Rename events
        #The turnover event includes just one player in the event information, meaning giveaways and takeaways will have no coordinates for play-by-plays created by ESPN scraping
        espn_events['event_type'] = espn_events['event_type'].replace({
            "Face Off":'faceoff',
            "Hit":'hit',
            "Shot":'shot-on-goal',
            "Missed":'missed-shot',
            "Blocked":'blocked-shot',
            "Goal":'goal',
            "Delayed Penalty":'delayed-penalty',
            "Penalty":'penalty'
        })

        #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
        espn_events['period_time_elapsed'] = espn_events['period_time_elapsed'].fillna('0:00')
        espn_events['period_seconds_elapsed'] = espn_events['period_time_elapsed'].apply(convert_to_seconds)
        espn_events['seconds_elapsed'] = ((espn_events['period']-1)*1200)+espn_events['period_seconds_elapsed']

        #Add event team data
        espn_events['event_team_abbr'] = espn_events['event_team_venue'].replace({
            "away":away,
            "home":home
        })
        
        #Add temporary game_id for coordinate adjustment
        espn_events['game_id'] = game_id

        #Coordinate adjustments:
        # x, y - Raw coordinates from JSON pbp
        # x_adj, y_adj - Adjusted coordinates configuring the away offensive zone to the left and the home offensive zone to the right
        #Some games (mostly preseason and all star games) do not include coordinates. 
        try:
            espn_events = adjust_coords(espn_events)
        except KeyError:
            print(f"No coordinates found for game ...")
        
            espn_events['x_adj'] = np.nan
            espn_events['y_adj'] = np.nan
            espn_events['event_distance'] = np.nan
            espn_events['event_angle'] = np.nan

        #Return: play-by-play events in supplied game from ESPN
        return espn_events

def assign_target(data):
    #Assign target number to plays to assist with merging

    #New sort
    data = data.sort_values(['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','event_player_2_id'])

    #Target number distingushes events that occur in the same second to assist in merging the JSON and HTML
    #Sometimes the target number may not reflect the same order as the event number in either document (especially in earlier seasons where the events are out of order in the HTML or JSON)
    data['target_num'] = np.where(data['event_type'].isin(['penalty','blocked-shot','missed-shot','shot-on-goal','goal']),data['event_type'].isin(['penalty','blocked-shot','missed-shot','shot-on-goal','goal']).cumsum(),0)

    #Revert sort and return dataframe
    return data.reset_index()

def combine_pbp(info,sources):
    #Given game info, return complete play-by-play data for provided game

    if info['season'] in [20052006, 20062007, 20072008, 20082009, 20092010]:
        #Create tasks
        html_task = parse_html(info)
        espn_task = parse_espn(str(info['game_date']),info['away_team_abbr'],info['home_team_abbr'])
        json_type = 'espn'
        json_task = parse_json(info)
    else:
        espn_task = None
        json_type = 'nhl'
        # NHL JSON + NHL HTML (HTML is required for descriptions / legacy compatibility).
        # Run in parallel since HTML fetch/parse is usually the slowest part.
        with ThreadPoolExecutor(max_workers=2) as executor:
            html_future = executor.submit(parse_html, info)
            json_future = executor.submit(parse_json, info)
            html_task = html_future.result()
            json_task = json_future.result()

    html_pbp = html_task
    json_pbp = json_task
    espn_pbp = espn_task
    
    #Route data combining - json if season is after 2009-2010:
    if json_type == 'espn':
        #ESPN x HTML
        espn_pbp = espn_pbp.sort_values(['period','seconds_elapsed']).reset_index()
        merge_col = ['period','seconds_elapsed','event_type','event_team_abbr']
        
        #Add additional information to espn_pbp with NHL json data
        espn_pbp = pd.merge(espn_pbp,json_pbp,how='left').drop_duplicates(subset='index')

        if sources:
            dirs_html = f'sources/{info['season']}/HTML/'
            dirs_json = f'sources/{info['season']}/JSON/'

            if not os.path.exists(dirs_html):
                os.makedirs(dirs_html)
            if not os.path.exists(dirs_json):
                os.makedirs(dirs_json)

            html_pbp.to_csv(f'{dirs_html}{info['game_id']}_HTML.csv',index=False)
            espn_pbp.to_csv(f'{dirs_json}{info['game_id']}_JSON.csv',index=False)

        print(f' merging on columns...',end="")
        #Merge pbp
        df = pd.merge(html_pbp,espn_pbp,how='left',on=merge_col)

    else:
        #JSON x HTML
        if sources:
            dirs_html = f'sources/{info['season']}/HTML/'
            dirs_json = f'sources/{info['season']}/JSON/'

            if not os.path.exists(dirs_html):
                os.makedirs(dirs_html)
            if not os.path.exists(dirs_json):
                os.makedirs(dirs_json)

            html_pbp.to_csv(f'{dirs_html}{info['game_id']}_HTML.csv',index=False)
            json_pbp.to_csv(f'{dirs_json}{info['game_id']}_JSON.csv',index=False)

        #Assign target numbers
        html_pbp = assign_target(html_pbp)
        json_pbp = assign_target(json_pbp)

        #Merge on index if the df lengths are the same and the events are in the same general order; merge on columns otherwise
        if (len(html_pbp) == len(json_pbp)) and (html_pbp['event_type'].equals(json_pbp['event_type'])) and (html_pbp['seconds_elapsed'].equals(json_pbp['seconds_elapsed'])):
            html_pbp = html_pbp.drop(columns=['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','event_player_2_id','event_player_3_id','shot_type','zone_code'],errors='ignore').reset_index()
            df = pd.merge(html_pbp,json_pbp,how='left',left_index=True,right_index=True).sort_values(['event_num'])
        else:
            print(f' merging on columns...',end="")
            #Modify merge conditions and merge pbps
            merge_col = ['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','target_num']
            html_pbp = html_pbp.drop(columns=['event_player_2_id','event_player_3_id','shot_type','zone_code'],errors='ignore')

            #While rare sometimes column 'event_player_1_id' is interpreted differently between the two dataframes. 
            html_pbp['event_player_1_id'] = html_pbp['event_player_1_id'].astype(object)
            json_pbp['event_player_1_id'] = json_pbp['event_player_1_id'].astype(object)

            #Merge pbp
            df = pd.merge(html_pbp,json_pbp,how='left',on=merge_col).sort_values(['event_num'])

    #Add game info
    info_col = ['season','season_type','game_id','game_date',"venue","venue_location",
        'away_team_abbr','home_team_abbr']
    
    for col in info_col:
        df[col] = info[col]

    #Fill period_type column and assign shifts a sub-500 event code
    df['period_type'] = np.where(df['period']<4,"REG",np.where(np.logical_and(df['period']==5,df['season_type']==2),"SO","OT"))
    try: df['event_type_code'] = np.where(df['event_type']!='change',df['event_type_code'],499)
    except:
        ""
    df = df.sort_values(['period','seconds_elapsed','event_num']).reset_index()

    df['event_team_venue'] = np.where(df['event_team_abbr'].isna(),"",np.where(df['home_team_abbr']==df['event_team_abbr'],"home","away"))
    
    # HTML parsing previously provided `strength_state`; if omitted, create it and let `combine_data()`
    # reconstruct the accurate value from shift events.
    if 'strength_state' not in df.columns:
        df['strength_state'] = ""

    if 'description' not in df.columns:
        df['description'] = ""

    #Correct strength state for penalty shots and shootouts - most games dont have shifts in shootout and are disculuded otherwise
    df['strength_state'] = np.where((df['period'].astype(str)=='5')&(df['event_type'].isin(['missed-shot','shot-on-goal','goal']))&(df['season_type']==2),"1v0",df['strength_state'])
    df['strength_state'] = np.where(df['description'].str.contains('Penalty Shot',case=False),"1v0",df['strength_state'])

    col = [col for col in PBP_COLS if col in df.columns.to_list()]
    #Return: complete play-by-play information for provided game
    return df[col]

## SHIFT SCRAPING FUNCTIONS ##
def parse_shifts_json(info):
    #Given game info, return json shift chart

    log = info['json_shifts']
    #Filter non-shift events and duplicate events
    log = log.loc[log['detailCode']==0].drop_duplicates(subset=['playerId','shiftNumber'])

    #Add full name columns
    log['player_name'] = (log['firstName'] + " " + log['lastName']).str.upper()

    log = log.rename(columns={
        'playerId':'player_id',
        'teamAbbrev':'event_team_abbr',
        'startTime':'start',
        'endTime':'end'
    })

    #Convert time columns
    log['start'] = log['start'].astype(str).apply(convert_to_seconds)
    log['end'] = log['end'].astype(str).apply(convert_to_seconds)
    log = log[['player_name','player_id',
                'period','event_team_abbr',
                'start','duration','end']]
    
    #Recalibrate duration
    log['duration'] = log['end'] - log['start']

    #Return: JSON shifts (seperated by team)
    away = log.loc[log['event_team_abbr']==info['away_team_abbr']]
    home = log.loc[log['event_team_abbr']==info['home_team_abbr']]

    return {'away':away,
            'home':home}

def analyze_shifts(shift, id, name, pos, team):
    #Collects teams in given shifts html (parsed by Beautiful Soup)
    #Modified version of Harry Shomer's analyze_shifts function in the hockey_scraper package
    shifts = {}

    shifts['player_name'] = name.upper()
    shifts['player_id'] = id
    shifts['player_pos'] = pos
    shifts['period'] = '4' if shift[1] == 'OT' else '5' if shift[1] == 'SO' else shift[1]
    shifts['event_team_abbr'] = get_team(team.strip(' '))
    shifts['start'] = convert_to_seconds(shift[2].split('/')[0])
    shifts['duration'] = convert_to_seconds(shift[4].split('/')[0])

    #Sometimes there are no digits
    if re.compile(r'\d+').findall(shift[3].split('/')[0]):
        shifts['end'] = convert_to_seconds(shift[3].split('/')[0])
    else:
        shifts['end'] = shifts['start'] + shifts['duration']
    return shifts

def parse_shifts_html(info,home):
    #Parsing of shifts data for a single team in a provided game
    #Modified version of Harry Shomer's parse_shifts function in the hockey_scraper package

    #Roster info prep
    roster = info['HTML_rosters']

    rosters = roster['home' if home else 'away']
    
    all_shifts = []
    #columns = ['game_id', 'player_name', 'player_id', 'period', 'team_abbr', 'start', 'end', 'duration']

    #Retreive HTML
    game_id = info['game_id']
    season = info['season']
    link = f"https://www.nhl.com/scores/htmlreports/{season}/T{'H' if home else 'V'}{game_id[-6:]}.HTM"
    doc = rs.get(link).content
    td, teams = get_soup(doc)

    team = teams[0]
    players = {}

    # Iterates through each player shifts table with the following data:
    # Shift #, Period, Start, End, and Duration.
    for t in td:
        t = t.get_text()
        if ',' in t and re.match(r'\d+', t):     # If a comma and number exists it is a player
            name = t
            
            name = name.split(',')
            number = int(name[0][:2].strip())
            #In very rare cases a player listed will be among the scratches for the same game.  
            #Keeping these is more likely than not misattribution
            try:
                id = rosters[str(number)][4]
                players[id] = {}

                #HTML shift functions assess one team at a time, which simplifies the lookup process with number to name and id
                
                players[id]['name'] = rosters[str(number)][2]
                players[id]['pos'] = rosters[str(number)][1]

                players[id]['shifts'] = []
            except KeyError:
                continue
        else:
            #If id somehow is not assigned at any point before this is ran then just skip
            try:
                #Pushes shifts to current player
                players[id]['shifts'].extend([t])
            except UnboundLocalError:
                continue

    for key in players.keys():
        # Create lists of shifts-table columns for analysis
        players[key]['shifts'] = [players[key]['shifts'][i:i + 5] for i in range(0, len(players[key]['shifts']), 5)]

        name = players[key]['name']
        pos = players[key]['pos']

        # Parsing
        shifts = [analyze_shifts(shift, key, name, pos, team) for shift in players[key]['shifts']]
        all_shifts.extend(shifts)

    df = pd.DataFrame(all_shifts)

    shifts_raw = df[df['duration'] > 0]

    #Return: single-team individual shifts by player
    return shifts_raw

def parse_shift_events(info,home):
    #Given game info and home team conditional, parse and convert document to shift events congruent to html play-by-play
    
    #Determine whether to use JSON shifts or HTML shifts
    if len(info['json_shifts']) == 0:
        shift = parse_shifts_html(info,home)
    else:
        shift = parse_shifts_json(info)['home' if home else 'away']

    rosters = info['rosters']

    # Identify shift starts for each shift event
    shifts_on = shift.groupby(['event_team_abbr', 'period', 'start']).agg(
        num_on=('player_name', 'size'),
        players_on=('player_name', lambda x: ', '.join(x)),
        ids_on=('player_id', lambda x: ', '.join(map(str,x))),
        ids_on_list=('player_id', lambda x: [str(v) for v in x]),
    ).reset_index()

    shifts_on = shifts_on.rename(columns={
        'start':"seconds_elapsed"
    })

    # Identify shift stops for each shift event
    shifts_off = shift.groupby(['event_team_abbr', 'period', 'end']).agg(
        num_off=('player_name', 'size'),
        players_off=('player_name', lambda x: ', '.join(x)),
        ids_off=('player_id', lambda x: ', '.join(map(str,x))),
        ids_off_list=('player_id', lambda x: [str(v) for v in x]),
    ).reset_index()

    shifts_off = shifts_off.rename(columns={
        'end':"seconds_elapsed"
    })

    # Merge and sort by time in game
    shifts = pd.merge(shifts_on, shifts_off, on=['event_team_abbr', 'period', 'seconds_elapsed'], how='outer')
    
    shifts['seconds_elapsed'] = shifts['seconds_elapsed'] + (1200*(shifts['period'].astype(int)-1))
    shifts['event_type'] = 'change'

    #Shift events similar to html (remove shootout shifts)
    shifts = shifts.loc[shifts['period'].astype(int)<5].sort_values(['period','seconds_elapsed']).reset_index(drop=True)

    #Generate on-ice columns (incremental set update; avoids O(players × events) regex scans)
    skater_ids_ordered = list(rosters.loc[rosters['positionCode']!="G",'playerId'].astype(str))
    goalie_ids_ordered = list(rosters.loc[rosters['positionCode']=="G",'playerId'].astype(str))
    team = shift['event_team_abbr'].iloc[0]

    ids_on_lists = shifts['ids_on_list'].tolist() if 'ids_on_list' in shifts else [None] * len(shifts)
    ids_off_lists = shifts['ids_off_list'].tolist() if 'ids_off_list' in shifts else [None] * len(shifts)

    on_ice: set[str] = set()
    prefix = 'home' if home else 'away'
    on_rows: list[dict[str, object]] = []

    for idx, (on_list, off_list) in enumerate(zip(ids_on_lists, ids_off_lists)):
        if isinstance(on_list, list):
            on_ice.update(on_list)
        if isinstance(off_list, list):
            for pid in off_list:
                on_ice.discard(pid)

        skaters_now = [pid for pid in skater_ids_ordered if pid in on_ice]
        goalies_now = [pid for pid in goalie_ids_ordered if pid in on_ice]

        row_out: dict[str, object] = {'row': idx}
        for i in range(6):
            row_out[f'{prefix}_on_{i+1}_id'] = skaters_now[i] if i < len(skaters_now) else " "

        # Empty net rows previously ended up as NaN -> "" after merge; preserve that by using "" here.
        row_out[f'{prefix}_goalie_id'] = goalies_now[0] if goalies_now else ""
        on_rows.append(row_out)

    on_players = pd.DataFrame(on_rows)

    shifts['row'] = shifts.index

    if home:
        shifts['home_team_abbr'] = team
    else:
        shifts['away_team_abbr'] = team
    #Return: shift events with newly added on-ice columns.  NAN values are replaced with string "REMOVE" as means to create proper on-ice columns for json pbp
    shifts = shifts.drop(columns=['ids_on_list','ids_off_list'], errors='ignore')
    return pd.merge(shifts,on_players,how="outer",on=['row']).replace(np.nan,"")

## FINALIZE PBP FUNCTIONS ##
def combine_shifts(info,sources):
    #Given game info, return complete shift events

    #JSON Prep
    roster = info['rosters']

    #Quickly combine shifts data (home/away independent; parallelize for speed, especially when falling back to HTML shifts)
    with ThreadPoolExecutor(max_workers=2) as executor:
        away_future = executor.submit(parse_shift_events, info, False)
        home_future = executor.submit(parse_shift_events, info, True)
        away = away_future.result()
        home = home_future.result()

    #Combine shifts
    data = pd.concat([away,home]).sort_values(['period','seconds_elapsed'])

    #Add game info
    info_col = ['season','season_type','game_id','game_date',"venue","venue_location",
        'away_team_abbr','home_team_abbr']
    
    for col in info_col:
        data[col] = info[col]

    #Create player information dicts to create on-ice names
    roster['playerId'] = roster['playerId'].astype(str)
    players = roster.set_index("playerId")['player_name'].to_dict()

    for i in range(0,7):
        if i == 6:
            data['away_goalie'] = data['away_goalie_id'].replace(players)
            data['home_goalie'] = data['home_goalie_id'].replace(players)
        else:
            data[f'away_on_{i+1}'] = data[f'away_on_{i+1}_id'].replace(players)
            data[f'home_on_{i+1}'] = data[f'home_on_{i+1}_id'].replace(players)

    data = data.sort_values(['period','seconds_elapsed'])
    #Fill on-ice columns down
    on_ice_col = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',
                'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',
                'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',
                'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',
                'away_goalie','home_goalie','away_goalie_id','home_goalie_id']

    for col in on_ice_col:
        data[col] = data[col].ffill()

    #Create strength state information
    away_on = ['away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id']
    home_on = ['home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id']
    data['away_skaters'] = data[away_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    data['home_skaters'] = data[home_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    data['strength_state'] = np.where(data['event_team_abbr']==data['away_team_abbr'],data['away_skaters'].astype(str)+"v"+data['home_skaters'].astype(str),data['home_skaters'].astype(str)+"v"+data['away_skaters'].astype(str))

    #Create final shifts df
    col = [col for col in PBP_COLS if col in data.columns.to_list()]
    full_shifts = data[col]
    
    #Export sources if true
    if sources:
        dirs = f'sources/{info['season']}/SHIFTS/'

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        full_shifts.to_csv(f'{dirs}{info['game_id']}_SHIFTS.csv',index=False)

    #Return: full shifts data converted to play-by-play format
    return full_shifts

def logical_sort(df):
    #Create priority columns designed to order events that occur at the same time in a game
    even_pri = ['takeaway','giveaway','missed-shot','hit','shot-on-goal','blocked-shot']
    df['priority'] = np.where(df['event_type'].isin(even_pri),1,
                              np.where(df['event_type']=='goal',2,
                              np.where(df['event_type']=='stoppage',3,
                              np.where(df['event_type']=='delayed-penalty',4,
                              np.where(df['event_type']=='penalty',5,
                              np.where(df['event_type']=='period-end',6,
                              np.where(df['event_type']=='change',7,
                              np.where(df['event_type']=='game-end',8,
                              np.where(df['event_type']=='period-start',9,
                              np.where(df['event_type']=='faceoff',10,0))))))))))

    df[['period','seconds_elapsed']] =  df[['period','seconds_elapsed']].astype(int)
    df = df.sort_values(['period','seconds_elapsed','event_num','priority'])

    return df

def combine_data(info,sources):
    #Given game info, return complete play-by-play data

    # PBP (HTML fetch/parse) and shifts are independent; run in parallel to reduce wall time.
    with ThreadPoolExecutor(max_workers=2) as executor:
        pbp_future = executor.submit(combine_pbp, info, sources)
        shifts_future = executor.submit(combine_shifts, info, sources)
        pbp = pbp_future.result()
        shifts = shifts_future.result()

    #Combine data    
    df = pd.concat([pbp,shifts])

    df['game_id'] = df['game_id'].astype(int)
    df['event_num'] = df['event_num'].replace(np.nan,0)

    df = logical_sort(df)
    
    #Recalibrate event_num column to accurately depict the order of all events, including changes
    df.reset_index(inplace=True,drop=True)
    df['event_num'] = df.index+1
    df['event_team_venue'] = np.where(df['event_team_abbr'].isna(),"",np.where(df['home_team_abbr']==df['event_team_abbr'],"home","away"))
    df['event_type_last'] = df['event_type'].shift(1)
    df['event_type_last_2'] = df['event_type_last'].shift(1)
    df['event_type_next'] = df['event_type'].shift(-1)
    lag_events = ['stoppage','goal','period-end']
    lead_events = ['faceoff','period-end']
    period_end_secs = [0,1200,2400,3600,4800,6000,7200,8400,9600,10800]
    
    #Define shifts by "line-change" or "on-the-fly"
    df['shift_type'] = np.where(df['event_type']=='change',np.where(np.logical_or(np.logical_or(df['event_type_last'].isin(lag_events),df['event_type_last_2'].isin(lag_events),df['event_type_next'].isin(lead_events)),df['seconds_elapsed'].isin(period_end_secs)),"line-change","on-the-fly"),"")
    df['description'] = df['description'].combine_first(df['event_team_abbr']+" CHANGE: "+df['shift_type'])
    try:
        df['event_type_code'] = np.where(df['event_type']=='change',499,df['event_type_code'])
    except:
        ""

    #Add time since last event to calculate the length of the prior event
    df['seconds_since_last'] = df['seconds_elapsed'] - df['seconds_elapsed'].shift(1)
    
    #Shootout attempts cannot contribute to TOI
    df['seconds_since_last'] = np.where(df['period_type']=='SO',0,df['seconds_since_last'])

    #Event length
    df['event_length'] = df['seconds_since_last'].shift(-1)

    #Add fixed strength state column
    df['strength_state_venue'] = df['away_skaters'].astype(str)+'v'+df['home_skaters'].astype(str)

    #Retrieve coaches
    coaches = info['coaches']
    if not coaches:
        df['away_coach'] = ""
        df['home_coach'] = ""
        df['event_coach'] = ""
    else:
        df['away_coach'] = coaches['away']
        df['home_coach'] = coaches['home']
        df['event_coach'] = np.where(df['event_team_abbr']==df['home_team_abbr'],coaches['home'],np.where(df['event_team_abbr']==df['away_team_abbr'],coaches['away'],""))

    #Fix event goalies
    df['event_goalie_id'] = np.where(df['event_team_venue']=='away',df['home_goalie_id'],df['away_goalie_id'])

    #Assign score, corsi, fenwick, and penalties for each event
    for venue in ['away','home']:
        df[f'{venue}_score'] = ((df['event_team_venue']==venue)&(df['event_type']=='goal')).cumsum().shift(1)
        df[f'{venue}_corsi'] = ((df['event_team_venue']==venue)&(df['event_type'].isin(['blocked-shot','missed-shot','shot-on-goal','goal']))).cumsum().shift(1)
        df[f'{venue}_fenwick'] = ((df['event_team_venue']==venue)&(df['event_type'].isin(['missed-shot','shot-on-goal','goal']))).cumsum().shift(1)
        df[f'{venue}_penalties'] = ((df['event_team_venue']==venue)&(df['event_type']=='penalty')).cumsum().shift(1)
    
    #Add time adjustments
    df['period_time'] = np.trunc((df['seconds_elapsed']-((df['period']-1)*1200))/60).astype(str).str.replace('.0','')+":"+(df['seconds_elapsed'] % 60).astype(str).str.pad(2,'left','0')
    df['game_time'] = np.trunc(df['seconds_elapsed']/60).astype(str).str.replace('.0','')+":"+(df['seconds_elapsed'] % 60).astype(str).str.pad(2,'left','0')

    #Forward fill as necessary
    cols = ['period_type','home_team_defending_side','away_coach','home_coach']
    for col in cols:
        try: df[col]
        except: df[col] = ""
        df[col] = df[col].ffill()
    
    #Add event player numbers
    roster = info['rosters']
    num_dict = roster.set_index('playerId')['sweaterNumber']
    
    for i in range(3):
        df[f'event_player_{i+1}_num'] = df[f'event_player_{i+1}_id'].map(num_dict)

    df['event_goalie_num'] = df['event_goalie_id'].map(num_dict)

    #Add on-ice player positions
    roster['playerId'] = roster['playerId'].astype(float)
    pos_dict = roster.set_index('playerId')['positionCode']
    for venue in ['away', 'home']:
        for i in range(6):
            if f'{venue}_on_{i+1}_id' in df.columns:
                df[f'{venue}_on_{i+1}_pos'] = df[f'{venue}_on_{i+1}_id'].replace(r'^\s*$', np.nan, regex=True).astype(float).map(pos_dict)
            else:
                df[f'{venue}_on_{i+1}_name'] = np.nan
                df[f'{venue}_on_{i+1}_id'] = np.nan
                df[f'{venue}_on_{i+1}_pos'] = np.nan

    #Return: complete play-by-play with all important data for each event in a provided game
    return df[[col for col in PBP_COLS if col in df.columns.to_list()]].replace(r'^\s*$', np.nan, regex=True)

## ROSTER FUNCTIONS ##
def parse_game_roster(rost_df, game_id):
    #Roster is already a dataframe so just standardize column names
    roster = rost_df.rename(columns=COL_MAP['roster'])

    #Add game id and season to link df to
    roster['game_id'] = game_id

    #Remove unwanted name columns
    roster = roster.drop(columns=[col for col in roster.columns if 'Name' in col])

    return roster
    
## NHL EDGE FUNCTIONS ##
def edge_stat_entry(entry, season, season_type, type):
    #Given entry (player or team id), season, season type, and type (player or team), return NHL Edge stats DataFrame

    def fetch_cat(cat):
        api = f'https://api-web.nhle.com/v1/edge/{type}-{cat}/{entry}/{season}/{season_type}'
        try:
            data = rs.get(api).json()
        except:
            return None

        edge = pd.json_normalize(data)
        edge['season'] = season

        # Zone-time expansion
        if cat in ('zone-time', 'zone-time-details') and 'zoneTimeDetails' in edge:
            zones = edge['zoneTimeDetails'].iloc[0]
            for zone in zones:
                strength = zone['strengthCode']
                for k, v in zone.items():
                    if k != 'strengthCode':
                        edge[f'{k}.{strength}'] = v

        return edge

    #Parallel fetch all categories
    dfs = []
    with ThreadPoolExecutor(max_workers=len(EDGE_CAT[type])) as executor:
        futures = [executor.submit(fetch_cat, cat) for cat in EDGE_CAT[type]]
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    #Merge all category DataFrames
    edge_df = dfs[0]
    for f in dfs[1:]:
        edge_df = pd.concat([edge_df, f.drop(columns='season', errors='ignore')], axis=1)

    if type != 'team':
        try:
            edge_df['player_name'] = (
                edge_df['player.firstName.default'] + " " + edge_df['player.lastName.default']
            ).str.upper()
        except KeyError:
            edge_df['player_name'] = ''

    #Return: NHL Edge stats DataFrame for entry provided
    return edge_df
