import json
import pandas as pd
import numpy as np
from wsba_hockey.tools.xg_model import *
from wsba_hockey.tools.globals import *

## AGGREGATE FUNCTIONS ##
# Provided in this file are functions vital to aggregating NHL play-by-play data into a wealth of statistics.

#Load globals
shot_types = SHOT_TYPES
fenwick_events = FENWICK_EVENTS
strengths_list = STRENGTH_MATCH
per_sixty = PER_SIXTY

def process_stats(df, group, venue, game_strength, second_group):
    #Determine columns
    team_col = f'{venue}_team_abbr'
    opp_col = f'{"home" if venue == "away" else "away"}_team_abbr'

    if group == 'skater':
        id_cols = [f'{venue}_on_{i}_id' for i in range(1, 7)]
        primary_cols = [f'{venue}_on_{i}_primary_fenwick_assist_probability' for i in range(1, 7)]
        secondary_cols = [f'{venue}_on_{i}_secondary_fenwick_assist_probability' for i in range(1, 7)]
        tertiary_cols = [f'{venue}_on_{i}_tertiary_fenwick_assist_probability' for i in range(1, 7)]
    elif group == 'goalie':
        id_cols = [f'{venue}_goalie_id']
    else:
        id_cols = []

    keep_cols = [
        'season','game_id','strength_state','event_num',
        team_col, opp_col, 'event_type', 'event_team_venue',
        'event_team_abbr','ids_on','shift_type','event_length',
        'zone_code','penalty_duration','penalty_type','short','xG'
    ]

    if group == 'skater':
        keep_cols += id_cols + primary_cols + secondary_cols + tertiary_cols
    elif group == 'goalie':
        keep_cols += id_cols

    df = df[keep_cols].copy()

    #Flip strength state (when necessary) and filter by game strength if not "all"
    if game_strength != 'all':
        df['strength_state_for'] = np.where(
            df['event_team_abbr'] == df[team_col],
            df['strength_state'],
            df['strength_state'].str[::-1]
        )
        df = df.loc[df['strength_state_for'].isin(game_strength)]

    if group in ['skater', 'goalie']:
        if group == 'skater':
            stacked = []

            for i in range(6):
                temp = df.copy()
                temp['player_id'] = temp[id_cols[i]]
                temp['primary_fenwick_assist_probability'] = temp[primary_cols[i]]
                temp['secondary_fenwick_assist_probability'] = temp[secondary_cols[i]]
                temp['tertiary_fenwick_assist_probability'] = temp[tertiary_cols[i]]
                stacked.append(temp)

            df = pd.concat(stacked, ignore_index=True)

            df = df.loc[df['player_id'] > 0]

        if group == 'goalie':
            df = df.rename(columns={id_cols[0]: 'player_id'})

    #Remove shots that don't reach net
    df['xG'] = np.where(df['short'] == 1, 0, df['xG'])
    valid_mask = df['short'] == 0

    event  = df['event_type']
    zone = df['zone_code']
    penalty_duration = df['penalty_duration']
    penalty_type = df['penalty_type']
    team_mask = df['event_team_abbr'] == df[team_col]
    opp_mask  = df['event_team_abbr'] == df[opp_col]

    df['expected_goals_for'] = ((team_mask & valid_mask) * df['xG']).astype(float)
    df['expected_goals_against'] = ((opp_mask & valid_mask) * df['xG']).astype(float)
    df['goals_for'] = ((event=='goal') & team_mask).astype(int)
    df['goals_against'] = ((event=='goal') & opp_mask).astype(int)
    df['shots_for'] = ((event.isin(['shot-on-goal','goal'])) & team_mask).astype(int)
    df['shots_against'] = ((event.isin(['shot-on-goal','goal'])) & opp_mask).astype(int)
    df['fenwick_for'] = ((event.isin(fenwick_events) & valid_mask) & team_mask).astype(int)
    df['fenwick_against'] = ((event.isin(fenwick_events) & valid_mask) & opp_mask).astype(int)
    df['corsi_for'] = ((event.isin(fenwick_events + ['blocked-shot'])) & team_mask).astype(int)
    df['corsi_against'] = ((event.isin(fenwick_events + ['blocked-shot'])) & opp_mask).astype(int)
    df['offensive_zone_faceoffs'] = ((event=='faceoff') & (((zone=='O') & team_mask) | ((zone=='D') & opp_mask))).astype(int)
    df['neutral_zone_faceoffs'] = ((event=='faceoff') & (((zone=='N') & team_mask) | ((zone=='N') & opp_mask))).astype(int)
    df['defensive_zone_faceoffs'] = ((event=='faceoff') & (((zone=='D') & team_mask) | ((zone=='O') & opp_mask))).astype(int)

    if group == 'skater':
        df['primary_fenwick_assists'] =  ((team_mask & valid_mask) * df['primary_fenwick_assist_probability']).astype(float)
        df['secondary_fenwick_assists'] = ((team_mask & valid_mask) * df['secondary_fenwick_assist_probability']).astype(float)
        df['tertiary_fenwick_assists'] = ((team_mask & valid_mask) * df['tertiary_fenwick_assist_probability']).astype(float)
        df['primary_expected_assists'] = (df['primary_fenwick_assists'] * df['xG']).astype(float)
        df['secondary_expected_assists'] = (df['secondary_fenwick_assists'] * df['xG']).astype(float)

    if group == 'team':
        df['hits_for'] = ((event=='hit') & team_mask).astype(int)
        df['hits_against'] = ((event=='hit') & opp_mask).astype(int)
        
        df['penalties'] = ((event=='penalty') & team_mask).astype(int)
        df['minor_penalties'] = ((event=='penalty') & (penalty_duration==2) & team_mask).astype(int)
        df['major_penalties'] = ((event=='penalty') & (penalty_duration==5) & team_mask).astype(int)
        df['fighting_penalties'] = ((event=='penalty') & (penalty_type=='fighting') & team_mask).astype(int)
        df['penalty_minutes'] = df.loc[(event=='penalty') & team_mask, 'penalty_duration']
        
        df['penalties_drawn'] = ((event=='penalty') & opp_mask).astype(int)
        df['minor_penalties_drawn'] = ((event=='penalty') & (penalty_duration==2) & opp_mask).astype(int)
        df['major_penalties_drawn'] = ((event=='penalty') & (penalty_duration==5) & opp_mask).astype(int)
        df['fighting_penalties_drawn'] = ((event=='penalty') & (penalty_type=='fighting') & opp_mask).astype(int)
        df['penalty_minutes_drawn'] = df.loc[(event=='penalty') & opp_mask, 'penalty_duration']
        
        df['giveaways'] = ((event=='giveaway') & team_mask).astype(int)
        df['takeaways'] = ((event=='takeaway') & team_mask).astype(int)
        df['blocked_shots'] = df['corsi_against']-df['fenwick_against']
    
    #Determine the primary grouping
    first_group = [team_col]
    if group in ['skater', 'goalie']:
        first_group.insert(0, 'player_id')
        
    agg_dict = {
        'games_played': ('game_id','nunique'),
        'time_on_ice': ('event_length','sum'),
        'fenwick_for': ('fenwick_for', 'sum'),
        'fenwick_against': ('fenwick_against', 'sum'),
        'goals_for': ('goals_for', 'sum'),
        'goals_against': ('goals_against', 'sum'),
        'shots_for': ('shots_for', 'sum'),
        'shots_against': ('shots_against', 'sum'),
        'expected_goals_for': ('expected_goals_for', 'sum'),
        'expected_goals_against': ('expected_goals_against', 'sum'),
        'corsi_for': ('corsi_for','sum'),
        'corsi_against': ('corsi_against','sum'),
        'offensive_zone_faceoffs': ('offensive_zone_faceoffs','sum'),
        'neutral_zone_faceoffs': ('neutral_zone_faceoffs','sum'),
        'defensive_zone_faceoffs': ('defensive_zone_faceoffs','sum')
    }

    if group == 'skater':
        agg_dict.update({
            'primary_fenwick_assists': ('primary_fenwick_assists', 'sum'),
            'secondary_fenwick_assists': ('secondary_fenwick_assists', 'sum'),
            'tertiary_fenwick_assists': ('tertiary_fenwick_assists', 'sum'),
            'primary_expected_assists': ('primary_expected_assists', 'sum'),
            'secondary_expected_assists': ('secondary_expected_assists', 'sum')
        })

    if group == 'team':
        agg_dict.update({
            'hits_for': ('hits_for','sum'),
            'hits_against': ('hits_against','sum'),
            'penalties': ('penalties','sum'),
            'minor_penalties': ('minor_penalties','sum'),
            'major_penalties': ('major_penalties','sum'),
            'fighting_penalties': ('fighting_penalties','sum'),
            'penalty_minutes': ('penalty_minutes','sum'),
            'penalties_drawn': ('penalties_drawn','sum'),
            'minor_penalties_drawn': ('minor_penalties_drawn','sum'),
            'major_penalties_drawn': ('major_penalties_drawn','sum'),
            'fighting_penalties_drawn': ('fighting_penalties_drawn','sum'),
            'penalty_minutes_drawn': ('penalty_minutes_drawn','sum'),
            'giveaways': ('giveaways','sum'),
            'takeaways': ('takeaways','sum'),
            'blocked_shots': ('blocked_shots','sum')
        })

    stats = df.groupby(first_group + second_group).agg(**agg_dict).reset_index()

    return stats.rename(columns={team_col:"team_abbr"}).drop(columns=[col for col in stats.columns if '_fenwick_assist_probability' in col])

def calc_indv(pbp,game_strength,second_group):
    # Filter by game strength if not "all"
    if game_strength != "all":
        pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]
        
    #Add second event-team column for necessary situations
    pbp['event_team_abbr_2'] = np.where(pbp['event_team_abbr'].notna(),
        np.where(pbp['event_team_abbr']==pbp['home_team_abbr'],pbp['away_team_abbr'],pbp['home_team_abbr']),np.nan)

    #Change second event team to goal-scoring team for goal events
    pbp['event_team_abbr_2'] = np.where(pbp['event_type']=='goal',pbp['event_team_abbr'],pbp['event_team_abbr_2'])

    #Determine how to group
    raw_group_1 = ['event_player_1_id','event_team_abbr']+second_group
    raw_group_2 = ['event_player_2_id','event_team_abbr_2']+second_group
    raw_group_3 = ['event_player_3_id','event_team_abbr']+second_group
    clean_group = ['player_id','team_abbr']+second_group

    #Add columns to sum on for player (if necessary)
    pbp['is_goal'] = (pbp['event_type'] == 'goal').astype(int)
    pbp['is_shot'] = pbp['event_type'].isin(['shot-on-goal','goal']).astype(int)
    pbp['is_fenwick'] = pbp['event_type'].isin(fenwick_events).astype(int)
    pbp['is_corsi'] = pbp['event_type'].isin(fenwick_events + ['blocked-shot']).astype(int)
    pbp['is_block'] = ((pbp['event_type'] == 'blocked-shot')&(pbp['event_reason']!='teammate-blocked')).astype(int)
    pbp['is_hit'] = (pbp['event_type'] == 'hit').astype(int)
    pbp['is_giveaway'] = (pbp['event_type'] == 'giveaway').astype(int)
    pbp['is_takeaway'] = (pbp['event_type'] == 'takeaway').astype(int)
    pbp['is_penalty'] = (pbp['event_type'] == 'penalty').astype(int)
    pbp['is_faceoff'] = (pbp['event_type'] == 'faceoff').astype(int)
    pbp['is_minor'] = (pbp['penalty_duration'] == 2).astype(int)
    pbp['is_major'] = (pbp['penalty_duration'] == 5).astype(int)
    pbp['is_fighting'] = (pbp['penalty_type'] == 'fighting').astype(int)

    #Remove shots that don't reach net
    for col in ['is_fenwick', 'xG']:
        pbp[col] = np.where(pbp['short']==1, 0, pbp[col])

    #Play-by-play to generate stats from
    agg_pbp = pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot","blocked-shot",'hit','giveaway','takeaway','faceoff','penalty'])]

    #First event player stats
    ep1 = (
        agg_pbp.groupby(raw_group_1).agg(
            goals=('is_goal', 'sum'),
            shots=('is_shot', 'sum'),
            fenwick=('is_fenwick', 'sum'),
            corsi=('is_corsi', 'sum'),
            expected_goals=('xG', 'sum'),
            hits_for=('is_hit', 'sum'),
            giveaways=('is_giveaway', 'sum'),
            takeaways=('is_takeaway', 'sum'),
            penalties=('is_penalty', 'sum'),
            minor_penalties=('is_minor', 'sum'),
            major_penalties=('is_major', 'sum'),
            fighting_penalties=('is_fighting', 'sum'),
            penalty_minutes=('penalty_duration','sum'),
            faceoff_wins=('is_faceoff', 'sum')
        )
    ).reset_index().rename(columns={'event_player_1_id': 'player_id', 'event_team_abbr': 'team_abbr'})

    #Second event player stats
    ep2 = (
        agg_pbp.groupby(raw_group_2).agg(
            primary_assists=('is_goal', 'sum'),
            hits_against=('is_hit', 'sum'),
            penalties_drawn=('is_penalty', 'sum'),
            minor_penalties_drawn=('is_minor', 'sum'),
            major_penalties_drawn=('is_major', 'sum'),
            fighting_penalties_drawn=('is_fighting', 'sum'),
            penalty_minutes_drawn=('penalty_duration', 'sum'),
            faceoff_losses=('is_faceoff', 'sum'),
            blocked_shots=('is_block', 'sum')
        )
    ).reset_index().rename(columns={'event_player_2_id': 'player_id', 'event_team_abbr_2': 'team_abbr'})

    #Third event player stats
    ep3 = (
        agg_pbp.groupby(raw_group_3).agg(
            secondary_assists=('is_goal', 'sum')
        )
    ).reset_index().rename(columns={'event_player_3_id': 'player_id', 'event_team_abbr': 'team_abbr'})
    
    indv = pd.merge(ep1,ep2,how='outer',on=clean_group)
    indv = pd.merge(indv,ep3,how='outer',on=clean_group)

    #Shot Types
    for st in shot_types:
        shot = (
            agg_pbp.loc[agg_pbp['shot_type']==st].groupby(raw_group_1).agg(
                goals=('is_goal', 'sum'),
                shots=('is_shot', 'sum'),
                fenwick=('is_fenwick', 'sum'),
                corsi=('is_corsi', 'sum'),
                expected_goals=('xG', 'sum'),
            )
        ).reset_index().rename(columns={'event_player_1_id': 'player_id', 'event_team_abbr': 'team_abbr'})

        st = st.replace('-','_')

        shot = shot.rename(columns={
            'goals':f'{st}_goals',
            'shots':f'{st}_shots',
            'fenwick':f'{st}_fenwick',
            'corsi':f'{st}_corsi',
            'expected_goals':f'{st}_expected_goals',
        })
        indv = pd.merge(indv,shot,how='outer',on=clean_group)

    indv[['goals','primary_assists','secondary_assists','penalties','penalties_drawn','faceoff_wins','faceoff_losses']] = indv[['goals','primary_assists','secondary_assists','penalties','penalties_drawn','faceoff_wins','faceoff_losses']].fillna(0)

    indv['primary_points'] = indv['goals'] + indv['primary_assists']
    indv['points'] = indv['primary_points'] + indv['secondary_assists']
    
    return indv

def calc_onice(pbp,game_strength,second_group):    
    home_stats = process_stats(pbp, 'skater', 'home', game_strength, second_group)
    away_stats = process_stats(pbp, 'skater', 'away', game_strength, second_group)

    onice_stats = pd.concat([home_stats, away_stats]).groupby(
        ['player_id','team_abbr','season'] + (['game_id'] if 'game_id' in second_group else [])
    ).agg(
        games_played=('games_played','sum'),
        time_on_ice=('time_on_ice','sum'),
        fenwick_for=('fenwick_for', 'sum'),
        fenwick_against=('fenwick_against', 'sum'),
        goals_for=('goals_for', 'sum'),
        goals_against=('goals_against', 'sum'),
        shots_for=('shots_for', 'sum'),
        shots_against=('shots_against', 'sum'),
        expected_goals_for=('expected_goals_for', 'sum'),
        expected_goals_against=('expected_goals_against', 'sum'),
        corsi_for=('corsi_for','sum'),
        corsi_against=('corsi_against','sum'),
        offensive_zone_faceoffs=('offensive_zone_faceoffs','sum'),
        neutral_zone_faceoffs=('neutral_zone_faceoffs','sum'),
        defensive_zone_faceoffs=('defensive_zone_faceoffs','sum'),
        primary_fenwick_assists=('primary_fenwick_assists', 'sum'),
        secondary_fenwick_assists=('secondary_fenwick_assists', 'sum'),
        tertiary_fenwick_assists=('tertiary_fenwick_assists', 'sum'),
        primary_expected_assists=('primary_expected_assists', 'sum'),
        secondary_expected_assists=('secondary_expected_assists', 'sum')
    ).reset_index()

    onice_stats['fenwick_assists'] = onice_stats['primary_fenwick_assists']+onice_stats['secondary_fenwick_assists']
    onice_stats['expected_assists'] = onice_stats['primary_expected_assists']+onice_stats['secondary_expected_assists']
    onice_stats['goals_saved_above_expected'] = onice_stats['expected_goals_against'] - onice_stats['goals_against']
    
    return onice_stats

def calc_team(pbp,game_strength,second_group):
    teams = []
    for venue in ['away', 'home']:
        stats = process_stats(
            pbp,
            'team',
            venue,
            game_strength,
            second_group
        )
        teams.append(stats)

    onice_stats = pd.concat(teams).groupby(
        ['team_abbr'] + second_group
    ).agg(
        games_played=('games_played','sum'),
        time_on_ice=('time_on_ice','sum'),
        fenwick_for=('fenwick_for', 'sum'),
        fenwick_against=('fenwick_against', 'sum'),
        goals_for=('goals_for', 'sum'),
        goals_against=('goals_against', 'sum'),
        shots_for=('shots_for','sum'),
        shots_against=('shots_against','sum'),
        expected_goals_for=('expected_goals_for', 'sum'),
        expected_goals_against=('expected_goals_against', 'sum'),
        corsi_for=('corsi_for','sum'),
        corsi_against=('corsi_against','sum'),
        offensive_zone_faceoffs=('offensive_zone_faceoffs','sum'),
        neutral_zone_faceoffs=('neutral_zone_faceoffs','sum'),
        defensive_zone_faceoffs=('defensive_zone_faceoffs','sum'),
        hits_for=('hits_for','sum'),
        hits_against=('hits_against','sum'),
        penalties=('penalties','sum'),
        minor_penalties=('minor_penalties','sum'),
        major_penalties=('major_penalties','sum'),
        fighting_penalties=('fighting_penalties','sum'),
        penalty_minutes=('penalty_minutes','sum'),
        penalties_drawn=('penalties_drawn','sum'),
        minor_penalties_drawn=('minor_penalties_drawn','sum'),
        major_penalties_drawn=('major_penalties_drawn','sum'),
        fighting_penalties_drawn=('fighting_penalties_drawn','sum'),
        penalty_minutes_drawn=('penalty_minutes_drawn','sum'),
        giveaways=('giveaways','sum'),
        takeaways=('takeaways','sum'),
        blocked_shots=('blocked_shots','sum')
    ).reset_index()

    onice_stats['goals_saved_above_expected'] = onice_stats['expected_goals_against'] - onice_stats['goals_against']

    return onice_stats

def calc_goalie(pbp,game_strength,second_group):
    teams = []
    for venue in ['away', 'home']:
        stats = process_stats(
            pbp,
            'goalie',
            venue,
            game_strength,
            second_group
        )
        teams.append(stats)

    onice_stats = pd.concat(teams).groupby(
        ['player_id','team_abbr'] + second_group
    ).agg(
        games_played=('games_played','sum'),
        time_on_ice=('time_on_ice','sum'),
        fenwick_for=('fenwick_for', 'sum'),
        fenwick_against=('fenwick_against', 'sum'),
        goals_for=('goals_for', 'sum'),
        goals_against=('goals_against', 'sum'),
        shots_for=('shots_for','sum'),
        shots_against=('shots_against','sum'),
        expected_goals_for=('expected_goals_for', 'sum'),
        expected_goals_against=('expected_goals_against', 'sum'),
        corsi_for=('corsi_for','sum'),
        corsi_against=('corsi_against','sum'),
        offensive_zone_faceoffs=('offensive_zone_faceoffs','sum'),
        neutral_zone_faceoffs=('neutral_zone_faceoffs','sum'),
        defensive_zone_faceoffs=('defensive_zone_faceoffs','sum')
    ).reset_index()

    onice_stats['goals_saved_above_expected'] = onice_stats['expected_goals_against'] - onice_stats['goals_against']
    
    return onice_stats

def rank_stats(df, rates=True, comparison=True, group_by=None):
    #Generate per sixty columns for raw totals and percentile columns for per sixty statss
    try:
        df['head_position'] = np.where(df['position'].isin(['C','L','R']), 'F', df['position'])
    except KeyError:
        df['head_position'] = 'team'

    #For obvious reasons, some columns in group_by can't be used to calculate percentiles
    if group_by:
        group_by = [col for col in group_by if col not in ['player_id', 'player_name', 'position', 'team_abbr']]
    else:
        group_by = []

    group_by.append('head_position')

    #Skip if no data from this function is desired
    if not rates and not comparison:
        return df.drop(columns=['head_position'])
    else:
        for stat in df.columns:
            if stat not in BIO_STAT_COL + ['player_id', 'season', 'time_on_ice', 'position', 'position_group'] \
            and pd.api.types.is_numeric_dtype(df[stat]):

                invert = (
                    'against' in stat
                    or ('penalties' in stat and 'drawn' not in stat)
                    or stat == 'giveaways'
                )

                #Ensure evaluated stats are raw totals
                if not any(s in stat for s in NON_TOTALS):
                    try:
                        if rates:
                            per_sixty = f'{stat}_per_sixty'
                            df[per_sixty] = (df[stat] / df['time_on_ice']) * 60
                        if comparison:
                            ranks = df.groupby(group_by)[per_sixty].rank(pct=True)
                            df[f'{per_sixty}_percentile'] = 1 - ranks if invert else ranks
                    except:
                        pass
                    
                if any(s in stat for s in SPECIAL_KEYS):
                    if comparison:
                        ranks = df.groupby(group_by)[stat].rank(pct=True)
                        df[f'{stat}_percentile'] = 1 - ranks if invert else ranks

        return df.drop(columns=['head_position'])

def extra_calc(df, metrics):
    #Calculate additional metrics with raw totals in an aggregated stats dataframe

    #Most dataframes will not include every column performed on so the metrics and their calculations will be checked and passed if it cannot be calculated with the available data
    #Metrics in "AGG_POST_METRICS" are organized in a tuple such as:
    #(METRIC, NUMERATOR OPERATION(S), DENOMINATOR OPERATION(S))
    for new_col, num_expr, denom_expr in AGG_POST_METRICS+metrics:
        try:
            num = df.eval(num_expr) if num_expr in df.columns or any(op in num_expr for op in '+-*/') else None
            denom = df.eval(denom_expr) if denom_expr in df.columns or (denom_expr and any(op in denom_expr for op in '+-*/')) else None

            if num is not None:
                if denom is not None:
                    df[new_col] = (num / denom.replace(0, np.nan)).fillna(0)
                else:
                    df[new_col] = num.fillna(0)

        except (KeyError, NameError):
            pass

    return df
    
def apply_rosters(df,group,schedule_path,roster_path):
    #Apply roster information to stats dataframe

    #Roster data for teams is result for each game
    if group == 'team':
        schedule = pd.read_csv(schedule_path)
        
        #Only want finished games for standing stats
        schedule = schedule.loc[~schedule['game_state'].isin(NON_FINAL_STATES)]

        #Add cumulative cols
        schedule['home_win'] = (schedule['home_score'] > schedule['away_score']).astype(int)
        schedule['away_win'] = (schedule['away_score'] > schedule['home_score']).astype(int)
        schedule['regulation'] = (schedule['period_type_last']=='REG').astype(int)
        schedule['overtime'] = (schedule['period_type_last']=='OT').astype(int)
        schedule['shootout'] = (schedule['period_type_last']=='SO').astype(int)

        dfs = []
        for venue in ['home','away']:
            standing = schedule[['game_id',f'{venue}_team_abbr',f'{venue}_win','regulation','overtime','shootout']].copy()
            for state in ['regulation','overtime','shootout']:
                standing[f'{state}_wins'] = np.where(standing[state]==1, standing[f'{venue}_win'], 0)
                standing[f'{state}_losses'] = np.where(standing[state]==1, 1-standing[f'{venue}_win'], 0)

            dfs.append(standing.drop(columns=[f'{venue}_win','regulation','overtime','shootout']).rename(columns={f'{venue}_team_abbr':'team_abbr'}))

        standing_stats = pd.concat(dfs)

        complete = pd.merge(df, standing_stats, how='left')

        #Add WSBA ID
        complete['wsba_id'] = complete['team_abbr']+complete['season'].astype(str)

    else:
        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['player_id','player_name',
                            'headshot','position','handedness',
                            'height_in','weight_lbs',
                            'birth_date','birth_country']].drop_duplicates(subset=['player_id'],keep='last')
        
        df['player_id'] = df['player_id'].astype(int)
        names['player_id'] = names['player_id'].astype(int)

        remove = [col for col in df.columns.intersection(names.columns) if col != 'player_id']
        
        #Add names
        complete = pd.merge(df.drop(remove, errors='ignore'),names,how='left')

        # Add player age
        complete['birth_date'] = pd.to_datetime(complete['birth_date'])
        complete['season_year'] = complete['season'].astype(str).str[4:8].astype(int)
        complete['age'] = complete['season_year'] - complete['birth_date'].dt.year

        # Find player headshot
        complete['headshot'] = 'https://assets.nhle.com/mugs/nhl/' + complete['season'].astype(str) + '/' + complete['team_abbr'].astype(str) + '/' + complete['player_id'].astype(int).astype(str) + '.png'

        #Add WSBA ID
        complete['wsba_id'] = complete['player_id'].astype(str).str.replace('.0','') + complete['season'].astype(str) + complete['team_abbr'].astype(str)

        # Remove goaltenders from skater dataframes
        if group == 'skater':
            complete = complete.loc[complete['position'] != 'G']
        elif group == 'game_score':
            complete = complete.loc[(complete['position'] != 'G') | ((complete['position'] == 'G') & (complete['points'].isna()))]
        
    #Return dataframe with stats info
    return complete

def apply_params(df, group_by, params, stage='before'):
    mask = pd.Series(True, index=df.index)

    for col, spec in params.items():
        op = spec[0]

        if spec[-1] in ('before', 'after'):
            timing = spec[-1]
            vals = spec[1:-1]
        else:
            timing = 'before'
            vals = spec[1:]

        if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
            vals = list(vals[0])

        if timing != stage:
            continue

        target_series = df[col]
        
        if op == 'last':
            n = vals[0]

            sort_cols = group_by + [col] if isinstance(group_by, list) else [group_by, col]
            df = df.sort_values(sort_cols)

            mask = (
                df.groupby(group_by, group_keys=False)[col]
                .transform(lambda s: pd.Series(
                    s.index.isin(s.tail(n).index),
                    index=s.index
                ))
            )

            df = df.loc[mask.fillna(False)]
        else:
            if 'date' in col.lower():
                target_series = pd.to_datetime(target_series, errors='coerce')
                vals = [pd.to_datetime(v) for v in vals]

            mask = OPS[op](target_series, *vals)
            df = df.loc[mask.fillna(False)]

    return df

def concat_col_values(s):
    #Numeric values will show range, strings will show all values

    is_num = pd.api.types.is_numeric_dtype(s)
    is_date = pd.api.types.is_datetime64_any_dtype(s)

    if (is_num or is_date) and s.name not in ['game_id', 'age', 'player_id', 'season_type']:
        s_min, s_max = s.min(), s.max()
        
        if s_min != s_max:
            if is_date:
                return f"{s_min.strftime('%Y-%m-%d')} - {s_max.strftime('%Y-%m-%d')}"
            return f"{s_min} - {s_max}"
        
    return ", ".join(pd.unique(s.astype(str)))

def sum_unique_games(series, games_df_ref, col_name):
    group_df = games_df_ref.loc[series.index]
    
    return group_df.drop_duplicates(subset='games_played')[col_name].sum()
