import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
import joblib

### XG_MODEL FUNCTIONS ###
# Provided in this file are functions vital to the goal prediction model in the WSBA Hockey Python package. #

def prep_xG_data(pbp):
    #Prep data for xG training and calculation

    events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal']
    shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    strengths = ['3v3',
                '3v4',
                '3v5',
                '4v3',
                '4v4',
                '4v5',
                '4v6',
                '5v3',
                '5v4',
                '5v5',
                '5v6',
                '6v4',
                '6v5']
    
    #Filter unwanted date:
    #Shots must occur in specified events and strength states, occur before the shootout, and have valid coordinates
    data = pbp.loc[(pbp['event_type'].isin(events))&
                   (pbp['strength_state'].isin(strengths))&
                   (pbp['period'] < 5)&
                   (pbp['x_fixed'].notna())&
                   (pbp['y_fixed'].notna())&
                   ~((pbp['x_fixed']==0)&(pbp['y_fixed']==0)&(pbp['x_fixed'].isin(fenwick_events))&(pbp['event_distance']!=90))]
    #Create last event columns
    data = data.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])

    data["seconds_since_last"] = data['seconds_elapsed']-data['seconds_elapsed'].shift(1)
    data["event_team_last"] = data['event_team_abbr'].shift(1)
    data["event_type_last"] = data['event_type'].shift(1)
    data["x_fixed_last"] = data['x_fixed'].shift(1)
    data["y_fixed_last"] = data['y_fixed'].shift(1)
    data["zone_code_last"] = data['zone_code'].shift(1)
    data['shot_type'] = data['shot_type'].fillna('wrist')
    

    data.sort_values(['season','game_id','period','seconds_elapsed','event_num'],inplace=True)
    data['score_state'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_score']-data['home_score'],data['home_score']-data['away_score'])
    data['fenwick_state'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_fenwick']-data['home_fenwick'],data['home_fenwick']-data['away_fenwick'])
    data['distance_from_last'] = np.sqrt((data['x_fixed'] - data['x_fixed_last'])**2 + (data['y_fixed'] - data['y_fixed_last'])**2)
    data['rush_mod'] = np.where((data['event_type'].isin(fenwick_events))&(data['zone_code_last'].isin(['N','D']))&(data['x_fixed']>25)&(data['seconds_since_last']<5),5-data['seconds_since_last'],0)
    data['rebound_mod'] = np.where((data['event_type'].isin(fenwick_events))&(data['event_type_last'].isin(fenwick_events))&(data['seconds_since_last']<3),3-data['seconds_since_last'],0)

    #Create boolean variables
    data["is_goal"]=(data['event_type']=='goal').astype(int)
    data["is_home"]=(data['home_team_abbr']==data['event_team_abbr']).astype(int)


    for shot in shot_types:
        data[shot] = (data['shot_type']==shot).astype(int)
    for strength in strengths:
        data[f'state_{strength}'] = (data['strength_state']==strength).astype(int)
    for event in events[0:len(events)-1]:
        data[f'prior_{event}_same'] = ((data['event_type_last']==event)&(data['event_team_last']==data['event_team_abbr'])).astype(int)
        data[f'prior_{event}_opp'] = ((data['event_type_last']==event)&(data['event_team_last']!=data['event_team_abbr'])).astype(int)
    
    #Return: pbp data prepared to train and calculate the xG model
    return data

def wsba_xG(pbp, train = False, overwrite = False, model_path = "tools/xg_model/wsba_xg.joblib", train_runs = 20, test_runs = 20):
    #Train and calculate the WSBA Expected Goals model
    
    target = "is_goal"
    continous = ['event_distance',
                'event_angle',
                'seconds_elapsed',
                'period',
                'x_fixed',
                'y_fixed',
                'x_fixed_last',
                'y_fixed_last',
                'distance_from_last',
                'seconds_since_last',
                'score_state',
                'fenwick_state',
                'rush_mod',
                'rebound_mod']
    boolean = ['is_home',
            'state_3v3',
            'state_3v4',
            'state_3v5',
            'state_4v3',
            'state_4v4',
            'state_4v5',
            'state_4v6',
            'state_5v3',
            'state_5v4',
            'state_5v5',
            'state_5v6',
            'state_6v4',
            'state_6v5',
            'wrist',
            'deflected',
            'tip-in',
            'slap',
            'backhand',
            'snap',
            'wrap-around',
            'poke',
            'bat',
            'cradle',
            'between-legs',
            'prior_shot-on-goal_same',
            'prior_missed-shot_same',
            'prior_blocked-shot_same',
            'prior_giveaway_same',
            'prior_takeaway_same',
            'prior_hit_same',
            'prior_shot-on-goal_opp',
            'prior_missed-shot_opp',
            'prior_blocked-shot_opp',
            'prior_giveaway_opp',
            'prior_takeaway_opp',
            'prior_hit_opp',
            'prior_faceoff']
    
    #Prep Data
    data = prep_xG_data(pbp)

    #Convert to sparse
    data_sparse = sp.csr_matrix(data[[target]+continous+boolean])

    #Target and Predictors
    is_goal_vect = data_sparse[:, 0].A
    predictors = data_sparse[:, 1:]

    #XGB DataModel
    xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect)

    if train == True:
        run_num = 
    else:
        print("No data to add yet...")
