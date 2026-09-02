import re
import warnings
import os
import time
import numpy as np
import polars as pl
import requests as rs
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import wsba_hockey.tools.http as http_tools
from wsba_hockey.tools.http import get as http_get, pooled_session
from wsba_hockey.tools.globals import (
    COL_MAP,
    DEFAULT_ROSTER,
    EDGE_CAT,
    FENWICK_EVENTS,
    PBP_COLS,
    POS_BASE_PROB,
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


def seconds_expr(column):
    #Parse a period clock with native Polars string expressions.
    value = pl.col(column).cast(pl.String).str.strip_chars()
    minutes = value.str.extract(r'^(\d+):(\d{1,2})$', 1).cast(pl.Float64, strict=False)
    seconds = value.str.extract(r'^(\d+):(\d{1,2})$', 2).cast(pl.Float64, strict=False)
    return (
        pl.when(value == '-16:0-')
        .then(pl.lit(1200.0))
        .when(minutes.is_not_null() & seconds.is_not_null())
        .then(minutes * 60 + seconds)
        .otherwise(None)
    )

def adjust_coords(pbp):
    # Given JSON or ESPN play-by-play, return coordinates normalized so the
    # away offensive zone is on the left and the home offensive zone is on
    # the right.  NHL JSON labels the home defending side on some rows. For
    # older/ESPN data, infer it from the first non-neutral faceoff per period;
    # faceoff zones are relative to the event team.
    group_cols = ['game_id', 'period']
    if 'x' not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None, dtype=pl.Float64).alias('x'))
    if 'y' not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None, dtype=pl.Float64).alias('y'))
    if 'zone_code' not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None, dtype=pl.String).alias('zone_code'))
    if 'event_team_venue' not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None, dtype=pl.String).alias('event_team_venue'))
    if 'home_team_defending_side' not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(None, dtype=pl.String).alias('home_team_defending_side'))

    side = pl.col('home_team_defending_side').cast(pl.String).str.to_lowercase().str.strip_chars()
    labeled_side = pl.when(side.is_in(['left', 'right'])).then(side).otherwise(None)
    raw_x = pl.col('x').cast(pl.Float64, strict=False)
    home_event = pl.col('event_team_venue') == 'home'
    defensive_faceoff = pl.col('zone_code') == 'D'
    same_side_as_home_defense = home_event == defensive_faceoff
    faceoff_side = (
        pl.when(raw_x > 0)
        .then(pl.when(same_side_as_home_defense).then(pl.lit('right')).otherwise(pl.lit('left')))
        .when(raw_x < 0)
        .then(pl.when(same_side_as_home_defense).then(pl.lit('left')).otherwise(pl.lit('right')))
        .otherwise(None)
    )

    pbp = pbp.with_columns(labeled_side.alias('_labeled_side')).with_columns(
        pl.col('_labeled_side').drop_nulls().first().over(group_cols).alias('_group_labeled_side')
    )
    inferred_sides = (
        pbp
        .filter(
            (pl.col('event_type') == 'faceoff')
            & pl.col('zone_code').is_in(['O', 'D'])
            & raw_x.is_not_null()
            & (raw_x != 0)
        )
        .with_columns(faceoff_side.alias('_faceoff_side'))
        .group_by(group_cols, maintain_order=True)
        .agg(pl.col('_faceoff_side').drop_nulls().first())
    )
    pbp = (
        pbp
        .join(inferred_sides, on=group_cols, how='left')
        .with_columns(
            pl.coalesce([
                pl.col('_group_labeled_side'),
                pl.col('_faceoff_side'),
            ]).alias('home_team_defending_side')
        )
        .with_columns([
            pl.when(pl.col('home_team_defending_side') == 'right').then(-pl.col('x')).otherwise(pl.col('x')).alias('x_adj'),
            pl.when(pl.col('home_team_defending_side') == 'right').then(-pl.col('y')).otherwise(pl.col('y')).alias('y_adj'),
        ])
        .with_columns([
            pl.when(pl.col('event_team_venue') == 'home').then(((89 - pl.col('x_adj')).pow(2) + pl.col('y_adj').pow(2)).sqrt()).otherwise(((-89 - pl.col('x_adj')).pow(2) + pl.col('y_adj').pow(2)).sqrt()).alias('event_distance'),
            pl.when(pl.col('event_team_venue') == 'home').then(pl.arctan2(pl.col('y_adj').abs(), (89 - pl.col('x_adj')).abs()).degrees()).otherwise(pl.arctan2(pl.col('y_adj').abs(), (-89 - pl.col('x_adj')).abs()).degrees()).alias('event_angle'),
        ])
        .with_columns([
            pl.col('x_adj').abs().alias('x_fixed'),
            pl.when(pl.col('x_adj') < 0).then(-pl.col('y_adj')).otherwise(pl.col('y_adj')).alias('y_fixed'),
        ])
        .drop(['_labeled_side', '_group_labeled_side', '_faceoff_side'])
    )

    return pbp


def fix_players(pbp):
    # Add/fix player info for shooters and goaltenders.
    try:
        find = pbp.filter(pl.col('event_type').is_in(FENWICK_EVENTS) & pl.col('event_player_1_hand').is_null()).get_column('event_player_1_id').cast(pl.Int64, strict=False).drop_nulls().unique().to_list()
    except Exception:
        find = pbp['event_player_1_id'].cast(pl.Int64, strict=False).drop_nulls().unique().to_list()

    roster = pl.read_csv(DEFAULT_ROSTER)
    roster = roster.filter(pl.col('player_id').is_in(find)).unique('player_id').select(['player_name', 'player_id', 'position', 'handedness'])

    if find:
        print('Adding player info to pbp...')

        player_ids = pl.col('event_player_1_id').cast(pl.Int64, strict=False)
        missing = pbp.filter(player_ids.is_in(find) & ~player_ids.is_in(roster.get_column('player_id'))).get_column('event_player_1_id').cast(pl.Int64, strict=False).unique().drop_nulls().to_list()
        if missing:
            from wsba_hockey.wsba_main import nhl_scrape_player_info

            add = nhl_scrape_player_info(missing).select(['player_name', 'player_id', 'handedness']).with_columns(pl.lit(None).alias('position'))
            roster = pl.concat([roster, add], how='diagonal_relaxed')

        hand_dict = dict(zip(roster['player_id'].cast(pl.Int64, strict=False).to_list(), roster['handedness'].to_list()))
        pbp = pbp.with_columns([
            pl.when(pl.col('event_team_venue') == 'away').then(pl.col('home_goalie_id')).otherwise(pl.col('away_goalie_id')).alias('event_goalie_id'),
            pl.col('event_player_1_id').cast(pl.Int64, strict=False).replace(hand_dict, default=None).alias('event_player_1_hand'),
        ])

    return pbp


def apply_passing_imputation(pbp):
    # Estimate player passing/setting impacts on shot attempts.
    goals = pl.col('event_type') == 'goal'
    non_goals = pl.col('event_type').is_in(FENWICK_EVENTS) & (pl.col('event_type') != 'goal')

    def update_masked(frame, mask_expr, columns, values):
        mask = frame.select(mask_expr).to_series().to_numpy()
        updates = []
        for index, column in enumerate(columns):
            current = frame.get_column(column).to_numpy().copy()
            current[mask] = values[:, index]
            updates.append(pl.Series(column, current))
        return frame.with_columns(updates)

    for venue in ['away', 'home']:
        team_mask = pl.col('event_team_venue') == venue
        player_cols = [f'{venue}_on_{j}_id' for j in range(1, 7)]
        pos_cols = [f'{venue}_on_{j}_pos' for j in range(1, 7)]
        prob_cols = {
            'primary': [f'{venue}_on_{j}_primary_fenwick_assist_probability' for j in range(1, 7)],
            'secondary': [f'{venue}_on_{j}_secondary_fenwick_assist_probability' for j in range(1, 7)],
            'tertiary': [f'{venue}_on_{j}_tertiary_fenwick_assist_probability' for j in range(1, 7)]
        }

        pbp = pbp.with_columns([
            pl.lit(0.0).alias(column)
            for columns in prob_cols.values()
            for column in columns
        ])

        goal_team_mask = team_mask & goals
        goal_assist_exprs = []
        for j in range(1, 7):
            player_col = f'{venue}_on_{j}_id'
            player_ids = pl.col(player_col).cast(pl.Float64, strict=False)
            is_primary_assist = player_ids == pl.col('event_player_2_id').cast(pl.Float64, strict=False)
            is_secondary_assist = player_ids == pl.col('event_player_3_id').cast(pl.Float64, strict=False)
            goal_assist_exprs.extend([
                pl.when(goal_team_mask & is_primary_assist).then(1.0).otherwise(pl.col(f'{venue}_on_{j}_primary_fenwick_assist_probability')).alias(f'{venue}_on_{j}_primary_fenwick_assist_probability'),
                pl.when(goal_team_mask & is_secondary_assist).then(1.0).otherwise(pl.col(f'{venue}_on_{j}_secondary_fenwick_assist_probability')).alias(f'{venue}_on_{j}_secondary_fenwick_assist_probability'),
            ])
        pbp = pbp.with_columns(goal_assist_exprs)

        if pbp.filter(goal_team_mask).height:
            goal_df = pbp.filter(goal_team_mask)
            goal_on_ice_ids = goal_df.select(player_cols).with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in player_cols]).to_numpy()
            goal_on_ice_pos = goal_df.select(pos_cols).with_columns([pl.col(c).fill_null('').cast(pl.String).str.to_uppercase() for c in pos_cols]).to_numpy()
            tertiary_probs_goals = np.vectorize(
                lambda pos: POS_BASE_PROB['tertiary'].get(pos, 0)
            )(goal_on_ice_pos)

            goal_player_1 = goal_df['event_player_1_id'].cast(pl.Float64, strict=False).to_numpy()[:, None]
            goal_player_2 = goal_df['event_player_2_id'].cast(pl.Float64, strict=False).to_numpy()[:, None]
            goal_player_3 = goal_df['event_player_3_id'].cast(pl.Float64, strict=False).to_numpy()[:, None]
            involved_mask = (
                (goal_on_ice_ids == goal_player_1) |
                (goal_on_ice_ids == goal_player_2) |
                (goal_on_ice_ids == goal_player_3)
            )
            tertiary_probs_goals[involved_mask] = 0
            tertiary_sums = tertiary_probs_goals.sum(axis=1, keepdims=True)
            tertiary_sums[tertiary_sums == 0] = 1
            tertiary_probs_goals = (tertiary_probs_goals / tertiary_sums) * 0.8
            pbp = update_masked(pbp, goal_team_mask, prob_cols['tertiary'], tertiary_probs_goals)

        non_goal_team_mask = team_mask & non_goals
        if pbp.filter(non_goal_team_mask).height:
            non_goal_df = pbp.filter(non_goal_team_mask)
            on_ice_ids = non_goal_df.select(player_cols).with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in player_cols]).to_numpy()
            on_ice_pos = non_goal_df.select(pos_cols).with_columns([pl.col(c).fill_null('').cast(pl.String).str.to_uppercase() for c in pos_cols]).to_numpy()
            shooter_ids = non_goal_df['event_player_1_id'].cast(pl.Float64, strict=False).to_numpy()[:, None]

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
                values = probs[assist_type]
                pbp = update_masked(pbp, non_goal_team_mask, prob_cols[assist_type], values)

    return pbp

## JSON FUNCTIONS ##
def get_game_roster(json):
    #Given raw json data, return game rosters
    roster = pl.json_normalize(json['rosterSpots']).with_columns((pl.col('firstName.default') + " " + pl.col('lastName.default')).str.to_uppercase().alias('player_name'))

    #Return: roster information
    return roster

def get_game_info(game_id, session=None):
    #Given game_id, return game information
    
    def _cached_get_json(session: rs.Session, url: str, cache_path: str, max_age_sec: int = 120):
        try:
            if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < max_age_sec):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

        data = session.get(url).json()
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data

    #Retrieve data (parallelize independent endpoints to reduce wall time)
    if session is None:
        session = pooled_session()
    api_pbp = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    api_shifts = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    api_right_rail = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"

    cache_root = os.environ.get("WSBA_HOCKEY_CACHE_DIR", ".wsba_cache")
    pbp_cache = os.path.join(cache_root, "api", "pbp", f"{game_id}.json")
    shifts_cache = os.path.join(cache_root, "api", "shiftcharts", f"{game_id}.json")
    rr_cache = os.path.join(cache_root, "api", "right_rail", f"{game_id}.json")

    with ThreadPoolExecutor(max_workers=http_tools.POOL_WORKERS) as executor:
        pbp_future = executor.submit(_cached_get_json, session, api_pbp, pbp_cache)
        shifts_future = executor.submit(_cached_get_json, session, api_shifts, shifts_cache)
        rr_future = executor.submit(_cached_get_json, session, api_right_rail, rr_cache)

        pbp_json = pbp_future.result()
        shifts_json = shifts_future.result()

        # Right-rail content is missing for some games; preserve prior behavior (empty dict)
        coaches = {}
        officials = {
            'referee_1': None,
            'referee_2': None,
            'linesman_1': None,
            'linesman_2': None
        }
        try:
            rr_json = rr_future.result()
            game_info = rr_json.get('gameInfo', {})
            def right_rail_name(entry):
                if not isinstance(entry, dict):
                    return None
                value = entry.get('default')
                if value is None and isinstance(entry.get('fullName'), dict):
                    value = entry['fullName'].get('default')
                return value.upper() if isinstance(value, str) else None

            away = right_rail_name(game_info.get('awayTeam', {}).get('headCoach'))
            home = right_rail_name(game_info.get('homeTeam', {}).get('headCoach'))
            if away and home:
                coaches = {'away': away, 'home': home}
            referees = game_info.get('referees', [])
            linesmen = game_info.get('linesmen', [])
            officials = {
                'referee_1': right_rail_name(referees[0]) if len(referees) > 0 else None,
                'referee_2': right_rail_name(referees[1]) if len(referees) > 1 else None,
                'linesman_1': right_rail_name(linesmen[0]) if len(linesmen) > 0 else None,
                'linesman_2': right_rail_name(linesmen[1]) if len(linesmen) > 1 else None
            }
        except Exception:
            coaches = {}
            officials = {
                'referee_1': None,
                'referee_2': None,
                'linesman_1': None,
                'linesman_2': None
            }

    #Provide explicit error for games which have not yet occured
    if pbp_json['gameState'] in ['FUT', 'PRE']:
        raise ValueError("Game has not yet occured.")

    #Games don't always have JSON shifts, for whatever reason
    json_shifts = pl.json_normalize(shifts_json.get('data', []), strict=False, infer_schema_length=None)
    if shifts_json.get('total', 0) == 0:
        json_shifts = pl.DataFrame()

    #Split information
    base = pl.json_normalize(pbp_json)
    row = base.row(0, named=True)
    game_id, season, season_type = row['id'], row['season'], row['gameType']
    game_date, game_state, start_time = row['gameDate'], row['gameState'], row['startTimeUTC']
    venue, venue_location = row['venue.default'], row['venueLocation.default']
    away_team_id, away_team_abbr = row['awayTeam.id'], row['awayTeam.abbrev']
    home_team_id, home_team_abbr = row['homeTeam.id'], row['homeTeam.abbrev']

    #Add roster
    roster = get_game_roster(pbp_json)
    #In the HTML parsing process, player are identified by a regex pattern (ABB #00 such as BOS #37) or number and name in the following format: #00 NAME (i.e. #37 BERGERON) so these are added as IDs of sorts.  
    team_map = {away_team_id: away_team_abbr, home_team_id: home_team_abbr}
    roster = roster.with_columns([
        ('#' + pl.col('sweaterNumber').cast(pl.String) + " " + pl.col('lastName.default').str.to_uppercase()).alias('descID'),
        pl.col('teamId').replace(team_map, default=None).alias('team_abbr'),
    ]).with_columns((pl.col('team_abbr') + " #" + pl.col('sweaterNumber').cast(pl.String)).alias('key'))

    #Create an additional roster dictionary for use with HTML parsing
    roster_dict = {'away':{}, 'home':{}}
    
    #Evaluate and add players by team
    for team in ['away','home']:
        abbr = (away_team_abbr if team == 'away' else home_team_abbr)
        rost = roster.filter(pl.col('team_abbr') == abbr)
        
        #Now iterate through team players
        for player,id,num,pos,team_abbr,key in rost.select(['player_name','playerId','sweaterNumber','positionCode','team_abbr','key']).iter_rows():
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
            'events':pl.json_normalize(pbp_json['plays']),
            'rosters':roster,
            'HTML_rosters':roster_dict,
            'coaches':coaches,
            'officials':officials,
            'json_shifts':json_shifts}

def parse_json(info):
    #Given game info, return JSON document

    #Retreive data
    events = info['events']

    #Return error if game is set in the future
    if info['game_state'] == 'FUT':
        game_id = info['id'][0]
        raise ValueError(f"Game {game_id} has not occured yet.")
    
    #Test columns
    cols = ['eventId', 'timeInPeriod', 'timeRemaining', 'situationCode', 'homeTeamDefendingSide', 'typeCode', 'typeDescKey', 'sortOrder', 'periodDescriptor.number', 'periodDescriptor.periodType', 'periodDescriptor.maxRegulationPeriods', 'details.eventOwnerTeamId', 'details.losingPlayerId', 'details.winningPlayerId', 'details.xCoord', 'details.yCoord', 'details.zoneCode', 'pptReplayUrl', 'details.shotType', 'details.scoringPlayerId', 'details.scoringPlayerTotal', 'details.assist1PlayerId', 'details.assist1PlayerTotal', 'details.assist2PlayerId', 'details.assist2PlayerTotal', 'details.goalieInNetId', 'details.awayScore', 'details.homeScore', 'details.highlightClipSharingUrl', 'details.highlightClipSharingUrlFr', 'details.highlightClip', 'details.highlightClipFr', 'details.discreteClip', 'details.discreteClipFr', 'details.shootingPlayerId', 'details.awaySOG', 'details.homeSOG', 'details.playerId', 'details.hittingPlayerId', 'details.hitteePlayerId', 'details.reason', 'details.typeCode', 'details.descKey', 'details.duration', 'details.servedByPlayerId', 'details.secondaryReason', 'details.blockingPlayerId', 'details.committedByPlayerId', 'details.drawnByPlayerId', 'game_id', 'season', 'season_type', 'game_date']

    events = events.with_columns([pl.lit('').alias(col) for col in cols if col not in events.columns])

    #Event_player_columns include players in a given set of events; the higher the number, the greater the importance the event player was to the play
    events = events.with_columns(pl.coalesce([pl.col('details.winningPlayerId'), pl.col('details.scoringPlayerId'), pl.col('details.shootingPlayerId'), pl.col('details.playerId'), pl.col('details.hittingPlayerId'), pl.col('details.committedByPlayerId')]).alias('event_player_1_id'))
        
    team_map = {info['away_team_id']: info['away_team_abbr'], info['home_team_id']: info['home_team_abbr']}
    events = events.with_columns([
        pl.coalesce([pl.col('details.losingPlayerId'), pl.col('details.assist1PlayerId'), pl.col('details.hitteePlayerId'), pl.col('details.drawnByPlayerId'), pl.col('details.blockingPlayerId')]).alias('event_player_2_id'),
        pl.col('details.assist2PlayerId').alias('event_player_3_id'),
        pl.when(pl.col('details.eventOwnerTeamId') == info['home_team_id']).then(pl.lit('home')).otherwise(pl.lit('away')).alias('event_team_venue'),
        pl.col('details.eventOwnerTeamId').replace(team_map, default=None).alias('event_team_abbr'),
    ])

    #Rename columns to follow WSBA naming conventions
    events = events.rename({
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
        events = events.drop(['x','y','event_team_venue','period_seconds_elapsed','game_id',
                                      'period_time_elapsed', 'shot_type', 'zone_code', 'event_player_1_id', 'event_player_2_id', 'event_player_3_id'], strict=False)
    else:
        try:
            events = adjust_coords(events)
        except KeyError:
            game_id = info['game_id'][0]
            print(f"No coordinates found for game {game_id}...")
            events = events.with_columns([pl.lit(None).alias(c) for c in ['x_adj','y_adj','event_distance','event_angle']])
        
    #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
    events = events.with_columns(
        seconds_expr('period_time_elasped').alias('period_seconds_elapsed')
    ).with_columns(
        ((pl.col('period') - 1) * 1200 + pl.col('period_seconds_elapsed')).alias('seconds_elapsed')
    ).filter(pl.col('event_type') != '')

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


def clean_html_pbp(info, session=None):
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
        html = http_get(doc, session=session).content
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

def parse_html(info, session=None):
    #Given game info, return HTML event data

    #Retreive game information and html events
    rosters = info['HTML_rosters']
    events = clean_html_pbp(info, session=session)

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
            events_dict['event_skaters'] = home_skaters if event_team == info['home_team_abbr'] else away_skaters

        event_rows.append(events_dict)

    # HTML event fields are heterogeneous across games (for example, an
    # attribution can be numeric early and ``EVG``/``PPG`` later).  Let
    # Polars inspect the complete event stream only for this mixed payload.
    data = pl.json_normalize(event_rows, strict=False, infer_schema_length=None)
    html_on_ice_cols = []
    for venue in ['away', 'home']:
        html_on_ice_cols.extend([f'{venue}_on_{i}' for i in range(1, 7)])
        html_on_ice_cols.extend([f'{venue}_on_{i}_id' for i in range(1, 7)])
        html_on_ice_cols.extend([f'{venue}_goalie', f'{venue}_goalie_id'])

    for col in html_on_ice_cols:
        if col not in data.columns:
            data = data.with_columns(pl.lit(" ").alias(col))
        else:
            data = data.with_columns(pl.col(col).fill_null(" ").alias(col))

    data = data.with_columns(pl.col('event_type').replace({
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
    }).alias('event_type'))
    
    #Return: parsed HTML pbp
    return data

### ESPN SCRAPING FUNCTIONS ###
def espn_game_id(date,away,home,session=None):
    #Given a date formatted as YYYY-MM-DD and teams, return game id from ESPN schedule
    date = date.replace("-","")

    #Retreive data
    api = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={date}"
    schedule = pl.json_normalize(http_get(api, session=session).json()['events'])

    #Create team abbreviation columns
    schedule = schedule.with_columns([
        pl.col('shortName').str.slice(0, 3).str.strip_chars().alias('away_team_abbr'),
        pl.col('shortName').str.slice(-3).str.strip_chars().alias('home_team_abbr'),
    ])
    
    #Modify team abbreviations as necessary
    schedule = schedule.with_columns([pl.col(c).replace({'LA': 'LAK', 'NJ': 'NJD', 'SJ': 'SJS', 'TB': 'TBL'}).alias(c) for c in ['away_team_abbr', 'home_team_abbr']])

    #Retreive game id
    game_id = schedule.filter((pl.col('away_team_abbr') == away) & (pl.col('home_team_abbr') == home)).get_column('id').item(0)

    #Return: ESPN game id
    return game_id

def parse_espn(date,away,home,session=None):
    #Given a date formatted as YYYY-MM-DD and teams, return game events from ESPN
    game_id = espn_game_id(date,away,home,session=session)
    
    #Hidden ESPN API endpoint (akin to the gamecenter/{game_id}/play-by-play NHL endpoint)
    url = f'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={game_id}'
    data = http_get(url, session=session).json()
    teams = data['boxscore']['teams']

    #Retreive plays
    espn_events = pl.json_normalize(data['plays']).rename({
        'period.number':'period',
        'clock.displayValue':'period_time_elapsed',
        'coordinate.x':'x',
        'coordinate.y':'y',
        'type.text':'event_type',
    })
    
    #Some games are missing plays on ESPN, for some reason
    if espn_events.is_empty():
        print(f"No coordinates found for game ...")
        return pl.DataFrame(columns=['period','seconds_elapsed','event_type','event_team_abbr'])
    else:
        #Retreive event team venue with team data (maintain the team abbreviation fill-in at the bottom)
        venue_map = {
            teams[0]['team']['id']: teams[0]['homeAway'],
            teams[1]['team']['id']: teams[1]['homeAway'],
        }
        espn_events = espn_events.with_columns(
            pl.col('team.id').replace(venue_map, default=None).alias('event_team_venue')
        )

        #Rename events
        #The turnover event includes just one player in the event information, meaning giveaways and takeaways will have no coordinates for play-by-plays created by ESPN scraping
        espn_events = espn_events.with_columns(pl.col('event_type').replace({
            "Face Off":'faceoff',
            "Hit":'hit',
            "Shot":'shot-on-goal',
            "Missed":'missed-shot',
            "Blocked":'blocked-shot',
            "Goal":'goal',
            "Delayed Penalty":'delayed-penalty',
            "Penalty":'penalty'
        }).alias('event_type'))

        #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
        espn_events = espn_events.with_columns(
            pl.col('period_time_elapsed').fill_null('0:00').alias('period_time_elapsed')
        ).with_columns(
            seconds_expr('period_time_elapsed').alias('period_seconds_elapsed')
        ).with_columns(
            ((pl.col('period') - 1) * 1200 + pl.col('period_seconds_elapsed')).alias('seconds_elapsed')
        )

        #Add event team data
        espn_events = espn_events.with_columns(pl.col('event_team_venue').replace({
            "away":away,
            "home":home
        }).alias('event_team_abbr'))
        
        #Add temporary game_id for coordinate adjustment
        espn_events = espn_events.with_columns(pl.lit(game_id).alias('game_id'))

        #Coordinate adjustments:
        # x, y - Raw coordinates from JSON pbp
        # x_adj, y_adj - Adjusted coordinates configuring the away offensive zone to the left and the home offensive zone to the right
        #Some games (mostly preseason and all star games) do not include coordinates. 
        try:
            espn_events = adjust_coords(espn_events)
        except KeyError:
            print(f"No coordinates found for game ...")
        
            espn_events = espn_events.with_columns([pl.lit(None).alias(c) for c in ['x_adj', 'y_adj', 'event_distance', 'event_angle']])

        #Return: play-by-play events in supplied game from ESPN
        return espn_events

def assign_target(data):
    #Assign target number to plays to assist with merging

    #New sort
    data = data.sort(['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','event_player_2_id'])

    #Target number distingushes events that occur in the same second to assist in merging the JSON and HTML
    #Sometimes the target number may not reflect the same order as the event number in either document (especially in earlier seasons where the events are out of order in the HTML or JSON)
    target = data['event_type'].is_in(['penalty','blocked-shot','missed-shot','shot-on-goal','goal'])
    data = data.with_columns(pl.when(target).then(target.cast(pl.Int64).cum_sum()).otherwise(0).alias('target_num'))

    #Revert sort and return dataframe
    return data.with_row_index('index')

def combine_pbp(info,sources,session=None):
    #Given game info, return complete play-by-play data for provided game

    if info['season'] in [20052006, 20062007, 20072008, 20082009, 20092010]:
        #Create tasks
        html_task = parse_html(info, session=session)
        espn_task = parse_espn(str(info['game_date']),info['away_team_abbr'],info['home_team_abbr'],session=session)
        json_type = 'espn'
        json_task = parse_json(info)
    else:
        espn_task = None
        json_type = 'nhl'
        # NHL JSON + NHL HTML (HTML is required for descriptions / legacy compatibility).
        # Run in parallel since HTML fetch/parse is usually the slowest part.
        with ThreadPoolExecutor(max_workers=http_tools.POOL_WORKERS) as executor:
            html_future = executor.submit(parse_html, info, session)
            json_future = executor.submit(parse_json, info)
            html_task = html_future.result()
            json_task = json_future.result()

    html_pbp = html_task
    json_pbp = json_task
    espn_pbp = espn_task
    
    #Route data combining - json if season is after 2009-2010:
    if json_type == 'espn':
        #ESPN x HTML
        espn_pbp = espn_pbp.sort(['period','seconds_elapsed']).with_row_index('index')
        merge_col = ['period','seconds_elapsed','event_type','event_team_abbr']
        
        #Add additional information to espn_pbp with NHL json data
        espn_pbp = espn_pbp.join(json_pbp, on=merge_col, how='left', suffix='_json').unique('index', keep='first')

        if sources:
            source_season = info['season']
            source_game_id = info['game_id']
            dirs_html = f"sources/{source_season}/HTML/"
            dirs_json = f"sources/{source_season}/JSON/"

            if not os.path.exists(dirs_html):
                os.makedirs(dirs_html)
            if not os.path.exists(dirs_json):
                os.makedirs(dirs_json)

            html_pbp.write_csv(f"{dirs_html}{source_game_id}_HTML.csv",index=False)
            espn_pbp.write_csv(f"{dirs_json}{source_game_id}_JSON.csv",index=False)

        print(f' merging on columns...',end="")
        #Merge pbp
        df = html_pbp.join(espn_pbp, on=merge_col, how='left', suffix='_espn')

    else:
        #JSON x HTML
        if sources:
            source_season = info['season']
            source_game_id = info['game_id']
            dirs_html = f"sources/{source_season}/HTML/"
            dirs_json = f"sources/{source_season}/JSON/"

            if not os.path.exists(dirs_html):
                os.makedirs(dirs_html)
            if not os.path.exists(dirs_json):
                os.makedirs(dirs_json)

            html_pbp.write_csv(f"{dirs_html}{source_game_id}_HTML.csv",index=False)
            json_pbp.write_csv(f"{dirs_json}{source_game_id}_JSON.csv",index=False)

        #Assign target numbers
        html_pbp = assign_target(html_pbp)
        json_pbp = assign_target(json_pbp)

        #Merge on index if the df lengths are the same and the events are in the same general order; merge on columns otherwise
        if (len(html_pbp) == len(json_pbp)) and html_pbp['event_type'].equals(json_pbp['event_type']) and html_pbp['seconds_elapsed'].equals(json_pbp['seconds_elapsed']):
            html_pbp = html_pbp.drop(['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','event_player_2_id','event_player_3_id','shot_type','zone_code'], strict=False)
            df = json_pbp.join(html_pbp, on='index', how='left', suffix='_html').sort('event_num')
        else:
            print(f' merging on columns...',end="")
            #Modify merge conditions and merge pbps
            merge_col = ['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id','target_num']
            html_pbp = html_pbp.drop(['event_player_2_id','event_player_3_id','shot_type','zone_code'], strict=False)

            #While rare sometimes column 'event_player_1_id' is interpreted differently between the two dataframes. 
            html_pbp = html_pbp.with_columns(pl.col('event_player_1_id').cast(pl.String).alias('event_player_1_id'))
            json_pbp = json_pbp.with_columns(pl.col('event_player_1_id').cast(pl.String).alias('event_player_1_id'))

            #Merge pbp
            df = html_pbp.join(json_pbp, how='left', on=merge_col, suffix='_json').sort('event_num')

    #Add game info
    info_col = ['season','season_type','game_id','game_date',"venue","venue_location",
        'away_team_abbr','home_team_abbr']
    
    df = df.with_columns([pl.lit(info[col]).alias(col) for col in info_col] + [pl.lit(value).alias(col) for col, value in info.get('officials', {}).items()])

    #Fill period_type column and assign shifts a sub-500 event code
    df = df.drop('index', strict=False).with_columns(pl.when(pl.col('period') < 4).then(pl.lit('REG')).otherwise(pl.when((pl.col('period') == 5) & (pl.col('season_type') == 2)).then(pl.lit('SO')).otherwise(pl.lit('OT'))).alias('period_type')).sort(['period','seconds_elapsed','event_num']).with_row_index('index')
    if 'event_type_code' in df.columns:
        df = df.with_columns(pl.when(pl.col('event_type') != 'change').then(pl.col('event_type_code')).otherwise(499).alias('event_type_code'))
    df = df.with_columns(pl.when(pl.col('event_team_abbr').is_null()).then(pl.lit('')).otherwise(pl.when(pl.col('home_team_abbr') == pl.col('event_team_abbr')).then(pl.lit('home')).otherwise(pl.lit('away'))).alias('event_team_venue'))
    
    # HTML parsing previously provided `strength_state`; if omitted, create it and let `combine_data()`
    # reconstruct the accurate value from shift events.
    if 'strength_state' not in df.columns:
        df = df.with_columns(pl.lit('').alias('strength_state'))

    if 'description' not in df.columns:
        df = df.with_columns(pl.lit('').alias('description'))

    #Correct strength state for penalty shots and shootouts - most games dont have shifts in shootout and are disculuded otherwise
    df = df.with_columns([
        pl.when((pl.col('period') == 5) & pl.col('event_type').is_in(['missed-shot','shot-on-goal','goal']) & (pl.col('season_type') == 2)).then(pl.lit('1v0')).otherwise(pl.col('strength_state')).alias('strength_state'),
    ]).with_columns(pl.when(pl.col('description').cast(pl.String).str.contains('(?i)Penalty Shot')).then(pl.lit('1v0')).otherwise(pl.col('strength_state')).alias('strength_state'))

    col = [col for col in PBP_COLS if col in df.columns]
    #Return: complete play-by-play information for provided game
    return df[col]

## SHIFT SCRAPING FUNCTIONS ##
def parse_shifts_json(info):
    #Given game info, return json shift chart

    log = info['json_shifts']
    #Filter non-shift events and duplicate events
    log = log.filter(pl.col('detailCode') == 0).unique(['playerId', 'shiftNumber'])

    #Add full name columns
    log = log.with_columns((pl.col('firstName') + " " + pl.col('lastName')).str.to_uppercase().alias('player_name'))

    log = log.rename({
        'playerId':'player_id',
        'teamAbbrev':'event_team_abbr',
        'startTime':'start',
        'endTime':'end'
    })

    #Convert time columns
    log = log.with_columns([
        seconds_expr('start').alias('start'),
        seconds_expr('end').alias('end'),
    ])
    log = log.select(['player_name','player_id',
                'period','event_team_abbr',
                'start','duration','end'])
    
    #Recalibrate duration
    log = log.with_columns((pl.col('end') - pl.col('start')).alias('duration'))

    #Return: JSON shifts (seperated by team)
    away = log.filter(pl.col('event_team_abbr') == info['away_team_abbr'])
    home = log.filter(pl.col('event_team_abbr') == info['home_team_abbr'])

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

def parse_shifts_html(info,home,session=None):
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
    doc = http_get(link, session=session).content
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

    df = pl.DataFrame(all_shifts)

    shifts_raw = df.filter(pl.col('duration') > 0)

    #Return: single-team individual shifts by player
    return shifts_raw

def parse_shift_events(info,home,session=None):
    #Given game info and home team conditional, parse and convert document to shift events congruent to html play-by-play
    
    #Determine whether to use JSON shifts or HTML shifts
    if len(info['json_shifts']) == 0:
        shift = parse_shifts_html(info,home,session=session)
    else:
        shift = parse_shifts_json(info)['home' if home else 'away']

    rosters = info['rosters']

    # Identify shift starts for each shift event
    shifts_on = shift.group_by(['event_team_abbr', 'period', 'start'], maintain_order=True).agg([
        pl.len().alias('num_on'), pl.col('player_name').str.join(', ').alias('players_on'),
        pl.col('player_id').cast(pl.String).str.join(', ').alias('ids_on'), pl.col('player_id').cast(pl.String).implode().alias('ids_on_list'),
    ])

    shifts_on = shifts_on.rename({
        'start':"seconds_elapsed"
    })

    # Identify shift stops for each shift event
    shifts_off = shift.group_by(['event_team_abbr', 'period', 'end'], maintain_order=True).agg([
        pl.len().alias('num_off'), pl.col('player_name').str.join(', ').alias('players_off'),
        pl.col('player_id').cast(pl.String).str.join(', ').alias('ids_off'), pl.col('player_id').cast(pl.String).implode().alias('ids_off_list'),
    ])

    shifts_off = shifts_off.rename({
        'end':"seconds_elapsed"
    })

    # Merge and sort by time in game
    shifts = shifts_on.join(shifts_off, on=['event_team_abbr', 'period', 'seconds_elapsed'], how='full', coalesce=True).with_columns([
        (pl.col('seconds_elapsed') + 1200 * (pl.col('period').cast(pl.Int64) - 1)).alias('seconds_elapsed'), pl.lit('change').alias('event_type')])

    #Shift events similar to html (remove shootout shifts)
    shifts = shifts.filter(pl.col('period').cast(pl.Int64) < 5).sort(['period','seconds_elapsed'])

    #Generate on-ice columns (incremental set update; avoids O(players × events) regex scans)
    skater_ids_ordered = rosters.filter(pl.col('positionCode') != 'G').get_column('playerId').cast(pl.String).to_list()
    goalie_ids_ordered = rosters.filter(pl.col('positionCode') == 'G').get_column('playerId').cast(pl.String).to_list()
    team = shift['event_team_abbr'][0]

    ids_on_lists = shifts['ids_on_list'].to_list() if 'ids_on_list' in shifts else [None] * len(shifts)
    ids_off_lists = shifts['ids_off_list'].to_list() if 'ids_off_list' in shifts else [None] * len(shifts)

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

    on_players = pl.DataFrame(on_rows)

    shifts = shifts.with_row_index('row')

    shifts = shifts.with_columns(pl.lit(team).alias('home_team_abbr' if home else 'away_team_abbr'))
    #Return: shift events with newly added on-ice columns.  NAN values are replaced with string "REMOVE" as means to create proper on-ice columns for json pbp
    shifts = shifts.drop(['ids_on_list','ids_off_list'], strict=False)
    merged = shifts.join(on_players, how='full', on=['row'], coalesce=True)
    return merged.with_columns([pl.col(c).fill_null('') for c in merged.columns if merged[c].dtype == pl.String])

## FINALIZE PBP FUNCTIONS ##
def combine_shifts(info,sources,session=None):
    #Given game info, return complete shift events

    #JSON Prep
    roster = info['rosters']

    #Quickly combine shifts data (home/away independent; parallelize for speed, especially when falling back to HTML shifts)
    with ThreadPoolExecutor(max_workers=http_tools.POOL_WORKERS) as executor:
        away_future = executor.submit(parse_shift_events, info, False, session)
        home_future = executor.submit(parse_shift_events, info, True, session)
        away = away_future.result()
        home = home_future.result()

    #Combine shifts
    data = pl.concat([away,home], how='diagonal_relaxed').sort(['period','seconds_elapsed'])

    #Add game info
    info_col = ['season','season_type','game_id','game_date',"venue","venue_location",
        'away_team_abbr','home_team_abbr']
    
    data = data.with_columns([pl.lit(info[col]).alias(col) for col in info_col])

    #Create player information dicts to create on-ice names
    players = dict(zip(roster['playerId'].cast(pl.String).to_list(), roster['player_name'].to_list()))

    data = data.with_columns(
        [
            pl.col(f'{venue}_on_{i}_id').replace(players, default=None).alias(f'{venue}_on_{i}')
            for venue in ('away', 'home')
            for i in range(1, 7)
        ] + [
            pl.col(f'{venue}_goalie_id').replace(players, default=None).alias(f'{venue}_goalie')
            for venue in ('away', 'home')
        ]
    )

    data = data.sort(['period','seconds_elapsed'])
    #Fill on-ice columns down
    on_ice_col = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',
                'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',
                'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',
                'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',
                'away_goalie','home_goalie','away_goalie_id','home_goalie_id']

    data = data.with_columns([pl.col(col).forward_fill().alias(col) for col in on_ice_col])

    #Create strength state information
    away_on = ['away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id']
    home_on = ['home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id']
    data = data.with_columns([
        pl.sum_horizontal([pl.col(c).cast(pl.String).str.strip_chars().ne('') for c in away_on]).alias('away_skaters'),
        pl.sum_horizontal([pl.col(c).cast(pl.String).str.strip_chars().ne('') for c in home_on]).alias('home_skaters'),
    ]).with_columns(pl.when(pl.col('event_team_abbr') == pl.col('away_team_abbr')).then(pl.col('away_skaters').cast(pl.String) + 'v' + pl.col('home_skaters').cast(pl.String)).otherwise(pl.col('home_skaters').cast(pl.String) + 'v' + pl.col('away_skaters').cast(pl.String)).alias('strength_state'))

    #Create final shifts df
    col = [col for col in PBP_COLS if col in data.columns]
    full_shifts = data.select(col)
    
    #Export sources if true
    if sources:
        source_season = info['season']
        source_game_id = info['game_id']
        dirs = f"sources/{source_season}/SHIFTS/"

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        full_shifts.write_csv(f"{dirs}{source_game_id}_SHIFTS.csv")

    #Return: full shifts data converted to play-by-play format
    return full_shifts

def logical_sort(df):
    #Create priority columns designed to order events that occur at the same time in a game
    even_pri = ['takeaway','giveaway','missed-shot','hit','shot-on-goal','blocked-shot']
    priorities = {event: 1 for event in even_pri} | {'goal': 2, 'stoppage': 3, 'delayed-penalty': 4, 'penalty': 5, 'period-end': 6, 'change': 7, 'game-end': 8, 'period-start': 9, 'faceoff': 10}
    df = df.with_columns([pl.col('event_type').replace(priorities, default=0).alias('priority'), pl.col('period').cast(pl.Int64), pl.col('seconds_elapsed').cast(pl.Int64)]).sort(['period','seconds_elapsed','event_num','priority'])

    return df

def combine_data(info,sources,session=None):
    #Given game info, return complete play-by-play data

    if session is None:
        session = pooled_session()

    # PBP (HTML fetch/parse) and shifts are independent; run in parallel to reduce wall time.
    with ThreadPoolExecutor(max_workers=http_tools.POOL_WORKERS) as executor:
        pbp_future = executor.submit(combine_pbp, info, sources, session)
        shifts_future = executor.submit(combine_shifts, info, sources, session)
        pbp = pbp_future.result()
        shifts = shifts_future.result()

    #Combine data    
    df = pl.concat([pbp,shifts], how='diagonal_relaxed').with_columns([pl.col('game_id').cast(pl.Int64), pl.col('event_num').fill_null(0)])

    # Some older games never contain a third event player.  Keep the public
    # event-player shape stable so optional player fields remain nullable.
    missing_event_player_cols = [
        pl.lit(None).alias(f'event_player_{player}_{field}')
        for player in range(1, 4)
        for field in ('name', 'id', 'pos')
        if f'event_player_{player}_{field}' not in df.columns
    ]
    if missing_event_player_cols:
        df = df.with_columns(missing_event_player_cols)

    df = logical_sort(df)
    
    #Recalibrate event_num column to accurately depict the order of all events, including changes
    df = df.with_row_index('__row').with_columns([
        (pl.col('__row') + 1).alias('event_num'),
        pl.when(pl.col('event_team_abbr').is_null()).then(pl.lit('')).otherwise(pl.when(pl.col('home_team_abbr') == pl.col('event_team_abbr')).then(pl.lit('home')).otherwise(pl.lit('away'))).alias('event_team_venue'),
        pl.col('event_type').shift(1).alias('event_type_last'), pl.col('event_type').shift(2).alias('event_type_last_2'), pl.col('event_type').shift(-1).alias('event_type_next'),
    ]).drop('__row')
    lag_events = ['stoppage','goal','period-end']
    lead_events = ['faceoff','period-end']
    period_end_secs = [0,1200,2400,3600,4800,6000,7200,8400,9600,10800]
    
    #Define shifts by "line-change" or "on-the-fly"
    df = df.with_columns(
        pl.when(pl.col('event_type') == 'change')
        .then(pl.when(
            pl.col('event_type_last').is_in(lag_events)
            | pl.col('event_type_last_2').is_in(lag_events)
            | pl.col('event_type_next').is_in(lead_events)
            | pl.col('seconds_elapsed').is_in(period_end_secs)
        ).then(pl.lit('line-change')).otherwise(pl.lit('on-the-fly')))
        .otherwise(pl.lit('')).alias('shift_type')
    ).with_columns(
        pl.coalesce([pl.col('description'), pl.col('event_team_abbr') + ' CHANGE: ' + pl.col('shift_type')]).alias('description')
    )
    if 'event_type_code' in df.columns:
        df = df.with_columns(pl.when(pl.col('event_type') == 'change').then(499).otherwise(pl.col('event_type_code')).alias('event_type_code'))

    #Add time since last event to calculate the length of the prior event
    df = df.with_columns(
        (pl.col('seconds_elapsed') - pl.col('seconds_elapsed').shift(1)).alias('seconds_since_last')
    ).with_columns(
        pl.when(pl.col('period_type') == 'SO').then(0).otherwise(pl.col('seconds_since_last')).alias('seconds_since_last')
    ).with_columns(
        pl.col('seconds_since_last').shift(-1).alias('event_length')
    )

    #Add fixed strength state column
    df = df.with_columns((pl.col('away_skaters').cast(pl.String) + 'v' + pl.col('home_skaters').cast(pl.String)).alias('strength_state_venue'))

    #Retrieve coaches
    coaches = info['coaches']
    if not coaches:
        df = df.with_columns([pl.lit('').alias(c) for c in ['away_coach','home_coach','event_coach']])
    else:
        df = df.with_columns([pl.lit(coaches['away']).alias('away_coach'), pl.lit(coaches['home']).alias('home_coach'), pl.when(pl.col('event_team_abbr') == pl.col('home_team_abbr')).then(pl.lit(coaches['home'])).otherwise(pl.when(pl.col('event_team_abbr') == pl.col('away_team_abbr')).then(pl.lit(coaches['away'])).otherwise(pl.lit(''))).alias('event_coach')])

    for col, value in info.get('officials', {}).items():
        df = df.with_columns(pl.lit(value).alias(col))

    #Fix event goalies
    df = df.with_columns(pl.when(pl.col('event_team_venue') == 'away').then(pl.col('home_goalie_id')).otherwise(pl.col('away_goalie_id')).alias('event_goalie_id'))

    #Assign score, corsi, fenwick, and penalties for each event
    score_exprs = []
    for venue in ['away','home']:
        score_exprs.extend([
            ((pl.col('event_team_venue') == venue) & (pl.col('event_type') == 'goal')).cast(pl.Int64).cum_sum().shift(1).fill_null(0).alias(f'{venue}_score'),
            ((pl.col('event_team_venue') == venue) & pl.col('event_type').is_in(['blocked-shot','missed-shot','shot-on-goal','goal'])).cast(pl.Int64).cum_sum().shift(1).fill_null(0).alias(f'{venue}_corsi'),
            ((pl.col('event_team_venue') == venue) & pl.col('event_type').is_in(['missed-shot','shot-on-goal','goal'])).cast(pl.Int64).cum_sum().shift(1).fill_null(0).alias(f'{venue}_fenwick'),
            ((pl.col('event_team_venue') == venue) & (pl.col('event_type') == 'penalty')).cast(pl.Int64).cum_sum().shift(1).fill_null(0).alias(f'{venue}_penalties'),
        ])
    df = df.with_columns(score_exprs)
    
    #Add time adjustments
    df = df.with_columns([
        ((pl.col('seconds_elapsed') - ((pl.col('period') - 1) * 1200)) // 60).cast(pl.String).str.replace(r'\.0$', '') .alias('_period_minutes'),
        (pl.col('seconds_elapsed') % 60).cast(pl.String).str.zfill(2).alias('_seconds'),
        (pl.col('seconds_elapsed') // 60).cast(pl.String).str.replace(r'\.0$', '').alias('_game_minutes'),
    ]).with_columns([
        (pl.col('_period_minutes') + ':' + pl.col('_seconds')).alias('period_time'),
        (pl.col('_game_minutes') + ':' + pl.col('_seconds')).alias('game_time'),
    ]).drop(['_period_minutes', '_seconds', '_game_minutes'])

    #Forward fill as necessary
    cols = ['period_type','home_team_defending_side','away_coach','home_coach']
    missing = [pl.lit('').alias(col) for col in cols if col not in df.columns]
    if missing:
        df = df.with_columns(missing)
    df = df.with_columns([pl.col(col).forward_fill().alias(col) for col in cols])
    
    #Add event player numbers
    roster = info['rosters']
    num_dict = dict(zip(roster['playerId'].cast(pl.String).to_list(), roster['sweaterNumber'].to_list()))
    
    df = df.with_columns([
        pl.col(f'event_player_{i+1}_id').cast(pl.String).replace(num_dict, default=None).alias(f'event_player_{i+1}_num')
        for i in range(3)
    ] + [
        pl.col('event_goalie_id').cast(pl.String).replace(num_dict, default=None).alias('event_goalie_num')
    ])

    #Add on-ice player positions
    pos_dict = dict(zip(roster['playerId'].cast(pl.Float64, strict=False).to_list(), roster['positionCode'].to_list()))
    position_exprs = []
    missing_player_exprs = []
    for venue in ['away', 'home']:
        for i in range(6):
            player_id_col = f'{venue}_on_{i+1}_id'
            if player_id_col in df.columns:
                position_exprs.append(
                    pl.col(player_id_col).cast(pl.String).str.strip_chars()
                    .cast(pl.Float64, strict=False).replace(pos_dict, default=None)
                    .alias(f'{venue}_on_{i+1}_pos')
                )
            else:
                missing_player_exprs.extend([
                    pl.lit(None).alias(f'{venue}_on_{i+1}_name'),
                    pl.lit(None).alias(player_id_col),
                    pl.lit(None).alias(f'{venue}_on_{i+1}_pos'),
                ])
    if missing_player_exprs:
        df = df.with_columns(missing_player_exprs)
    if position_exprs:
        df = df.with_columns(position_exprs)

    #Return: complete play-by-play with all important data for each event in a provided game
    df = apply_passing_imputation(df)

    return df.select([col for col in PBP_COLS if col in df.columns]).with_columns([pl.when(pl.col(c).cast(pl.String).str.strip_chars() == '').then(None).otherwise(pl.col(c)).alias(c) for c in df.select([col for col in PBP_COLS if col in df.columns]).columns])

## ROSTER FUNCTIONS ##
def parse_game_roster(rost_df, game_id):
    #Roster is already a dataframe so just standardize column names
    roster = rost_df.rename(COL_MAP['roster'], strict=False)

    #Add game id and season to link df to
    roster = roster.with_columns(pl.lit(game_id).alias('game_id'))

    #Remove unwanted name columns
    roster = roster.drop([col for col in roster.columns if 'Name' in col], strict=False)

    return roster
    
## NHL EDGE FUNCTIONS ##
def edge_stat_entry(entry, season, season_type, type, session=None):
    #Given entry (player or team id), season, season type, and type (player or team), return NHL Edge stats DataFrame

    def fetch_cat(cat):
        api = f'https://api-web.nhle.com/v1/edge/{type}-{cat}/{entry}/{season}/{season_type}'
        try:
            data = http_get(api, session=session).json()
        except:
            return None

        edge = pl.json_normalize(data).with_columns(pl.lit(season).alias('season'))

        # Zone-time expansion
        if cat in ('zone-time', 'zone-time-details') and 'zoneTimeDetails' in edge:
            zones = edge['zoneTimeDetails'][0]
            for zone in zones:
                strength = zone['strengthCode']
                for k, v in zone.items():
                    if k != 'strengthCode':
                        edge = edge.with_columns(pl.lit(v).alias(f'{k}.{strength}'))

        return edge

    #Parallel fetch all categories
    dfs = []
    with ThreadPoolExecutor(max_workers=min(http_tools.POOL_WORKERS, len(EDGE_CAT[type]))) as executor:
        futures = [executor.submit(fetch_cat, cat) for cat in EDGE_CAT[type]]
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.is_empty():
                dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    #Merge all category DataFrames
    edge_df = dfs[0]
    for f in dfs[1:]:
        edge_df = pl.concat([edge_df, f.drop('season', strict=False)], how='horizontal')

    if type != 'team':
        try:
            edge_df = edge_df.with_columns((pl.col('player.firstName.default') + " " + pl.col('player.lastName.default')).str.to_uppercase().alias('player_name'))
        except KeyError:
            edge_df = edge_df.with_columns(pl.lit('').alias('player_name'))

    #Return: NHL Edge stats DataFrame for entry provided
    return edge_df

def parse_event_sprite(frames: list[dict], home_team_id, away_team_id) -> pl.DataFrame:
    #Given frames from an event, return scaled frame-by-frame data
    if isinstance(frames, dict):
        frames = frames.get('frames') or frames.get('data') or [frames]
    rows = []
    for frame in frames:
        row = {}
        for key, value in frame.items():
            if key != 'onIce':
                name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', key)
                row[re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()] = value

        players = {'home': [], 'away': [], 'unknown': []}
        puck = None
        for track_id, obj in (frame.get('onIce') or {}).items():
            if obj.get('id') == 1 or str(track_id) == '1' or not obj.get('playerId'):
                puck = obj
                continue

            team_id = obj.get('teamId')
            if team_id == home_team_id:
                venue = 'home'
            elif team_id == away_team_id:
                venue = 'away'
            elif str(track_id).startswith('6'):
                venue = 'home'
            elif str(track_id).startswith('7'):
                venue = 'away'
            else:
                venue = 'unknown'
            players[venue].append((str(track_id), obj))

        for venue, items in players.items():
            for index, (_, obj) in enumerate(sorted(items, key=lambda item: item[0]), start=1):
                for attr, value in obj.items():
                    if attr == 'id':
                        suffix = 'tracking_id'
                    elif attr == 'playerId':
                        suffix = 'id'
                    elif attr == 'teamAbbrev':
                        suffix = 'team_abbr'
                    else:
                        name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', attr)
                        suffix = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
                    row[f'{venue}_on_ice_{index}_{suffix}'] = value

        if puck is not None:
            for attr, value in puck.items():
                if attr in ('sweaterNumber', 'teamAbbrev', 'teamId'):
                    continue
                if attr == 'id':
                    suffix = 'tracking_id'
                elif attr == 'playerId':
                    suffix = 'id'
                else:
                    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', attr)
                    suffix = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
                row[f'puck_{suffix}'] = value

        rows.append(row)

    #Convert coordinates from the animation scale to the scale observed in PBP data
    df = pl.DataFrame(rows)
    for col in df.columns:
        if col.endswith('_x') and df[col].dtype.is_numeric():
            df = df.with_columns(((pl.col(col) / 2400.0) * 200.0 - 100.0).alias(col))
        elif col.endswith('_y') and df[col].dtype.is_numeric():
            df = df.with_columns(((pl.col(col) / 1020.0) * 84.0 - 42.0).alias(col))

    #Return: DataFrame with frame-by-frame event data
    return df
