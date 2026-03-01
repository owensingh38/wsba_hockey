import os
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
import matplotlib.pyplot as plt
import wsba_hockey.wsba_main as wsba
from typing import Literal
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

### XG_MODEL FUNCTIONS ###
# Provided in this file are functions vital to the goal prediction model in the WSBA Hockey Python package. #

def fix_players(pbp):
    #Add/fix player info for shooters and goaltenders

    #Find players that don't have a handness
    try:
        find = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_player_1_hand'].isna()),'event_player_1_id'].drop_duplicates().to_list()
    except:
        #If this fails then the column doesn't exist and all players are included in the search
        find = pbp['event_player_1_id'].drop_duplicates().to_list()
    
    roster = pd.read_csv(wsba.DEFAULT_ROSTER)
    roster = roster.loc[roster['player_id'].isin(find)].drop_duplicates(['player_id'])[['player_name','player_id','position','handedness']]

    #Load roster and locate players
    if not find:
        pass
    else:
        print('Adding player info to pbp...')

        #Some players are missing from the roster file (generally in newer seasons); add these manually
        miss = pbp.loc[((pbp['event_player_1_id'].isin(find))&(~pbp['event_player_1_id'].isin(roster['player_id']))),'event_player_1_id'].drop_duplicates().dropna().to_list()
        if miss:
            add = wsba.nhl_scrape_player_info(miss)[['player_name','player_id','handedness']]
            roster = pd.concat([roster,add]).reset_index(drop=True)

        #Conversion dict
        roster['player_id'] = roster['player_id'].astype('Int64')
        hand_dict = roster.set_index('player_id').to_dict()['handedness']

        #Fix event goalies
        pbp['event_goalie_id'] = np.where(pbp['event_team_venue']=='away',pbp['home_goalie_id'],pbp['away_goalie_id'])
            
        #Add hands
        pbp['event_player_1_hand'] = pbp['event_player_1_id'].astype('Int64').map(hand_dict).where(lambda x: x.notna(), None)
    
    return pbp

def apply_passing_imputation(pbp):
    #Apply the imputation scheme to estimate player passing (or setting) impacts on shot attempts
    #The process is described by Micah Blake McCurdy in his xG model writeup
    #https://hockeyviz.com/txt/xg7#setterImputation

    goals = pbp['event_type'] == 'goal'
    non_goals = pbp['event_type'].isin(wsba.FENWICK_EVENTS) & (pbp['event_type'] != 'goal')

    #Iterate through each venue and apply to venue on ice players
    for venue in ['away', 'home']:
        team_mask = pbp['event_team_venue'] == venue
        
        # Get all on-ice player and position columns at once
        player_cols = [f'{venue}_on_{j}_id' for j in range(1, 7)]
        pos_cols = [f'{venue}_on_{j}_pos' for j in range(1, 7)]
        
        # Initialize all probability columns at once
        prob_cols = {
            'primary': [f'{venue}_on_{j}_primary_fenwick_assist_probability' for j in range(1, 7)],
            'secondary': [f'{venue}_on_{j}_secondary_fenwick_assist_probability' for j in range(1, 7)],
            'tertiary': [f'{venue}_on_{j}_tertiary_fenwick_assist_probability' for j in range(1, 7)]
        }
        
        for assist_type in prob_cols:
            pbp[prob_cols[assist_type]] = 0.0
        
        # GOALS: Set known assist probabilities
        goal_team_mask = team_mask & goals
        
        # Vectorized assignment for goals
        for j in range(1, 7):
            player_col = f'{venue}_on_{j}_id'
            is_scorer = pbp[player_col] == pbp['event_player_1_id']
            is_primary_assist = pbp[player_col] == pbp['event_player_2_id']
            is_secondary_assist = pbp[player_col] == pbp['event_player_3_id']
            
            pbp.loc[goal_team_mask & is_primary_assist, f'{venue}_on_{j}_primary_fenwick_assist_probability'] = 1.0
            pbp.loc[goal_team_mask & is_secondary_assist, f'{venue}_on_{j}_secondary_fenwick_assist_probability'] = 1.0
        
        # GOALS: Compute tertiary assist probabilities
        if goal_team_mask.any():
            goal_indices = pbp.index[goal_team_mask]
            goal_on_ice_ids = pbp.loc[goal_indices, player_cols].values
            goal_on_ice_pos = pbp.loc[goal_indices, pos_cols].values
            
            # Map positions to tertiary probabilities
            tertiary_probs_goals = np.vectorize(lambda pos: wsba.POS_BASE_PROB['tertiary'].get(pos, 0))(goal_on_ice_pos)
            
            # Zero out scorer and assist players
            goal_player_1 = pbp.loc[goal_indices, 'event_player_1_id'].values[:, None]
            goal_player_2 = pbp.loc[goal_indices, 'event_player_2_id'].values[:, None]
            goal_player_3 = pbp.loc[goal_indices, 'event_player_3_id'].values[:, None]
            
            involved_mask = (
                (goal_on_ice_ids == goal_player_1) |
                (goal_on_ice_ids == goal_player_2) |
                (goal_on_ice_ids == goal_player_3)
            )
            tertiary_probs_goals[involved_mask] = 0
            
            # Normalize
            tertiary_sums = tertiary_probs_goals.sum(axis=1, keepdims=True)
            tertiary_sums[tertiary_sums == 0] = 1
            tertiary_probs_goals = (tertiary_probs_goals / tertiary_sums) * 0.8
            
            # Assign back
            pbp.loc[goal_indices, prob_cols['tertiary']] = tertiary_probs_goals
        
        # NON-GOALS: Compute all assist probabilities
        if non_goals.any():
            non_goal_team_mask = team_mask & non_goals
            non_goal_indices = pbp.index[non_goal_team_mask]
            
            on_ice_ids = pbp.loc[non_goal_indices, player_cols].values
            on_ice_pos = pbp.loc[non_goal_indices, pos_cols].values
            shooter_ids = pbp.loc[non_goal_indices, 'event_player_1_id'].values[:, None]
            
            # Map positions to probabilities for all assist types
            probs = {}
            for assist_type in ['primary', 'secondary', 'tertiary']:
                probs[assist_type] = np.vectorize(lambda pos: wsba.POS_BASE_PROB[assist_type].get(pos, 0))(on_ice_pos)
                # Zero out shooter
                probs[assist_type][on_ice_ids == shooter_ids] = 0
                # Normalize
                sums = probs[assist_type].sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                probs[assist_type] = (probs[assist_type] / sums) * 0.8
            
            # Assign all probabilities at once
            for assist_type in ['primary', 'secondary', 'tertiary']:
                pbp.loc[non_goal_indices, prob_cols[assist_type]] = probs[assist_type]
    
    return pbp

def apply_time_on_ice(pbp):
    #Apply time on ice (in shift) for all on-ice skaters and first event player (shooter)

    #Collect on ice columns
    home_cols = [f'home_on_{i}_id' for i in range(1, 7)]
    away_cols = [f'away_on_{i}_id' for i in range(1, 7)]
    on_ice_cols = home_cols + away_cols

    #Flatten on ice player columns
    players = pd.unique(pbp[on_ice_cols].values.ravel())
    players = players[~pd.isna(players)]

    is_change = pbp['event_type'] == 'change'
    ids_on  = pbp['ids_on'].fillna('').astype(str)
    ids_off = pbp['ids_off'].fillna('').astype(str)
    event_time = pbp['event_length'].fillna(0)

    #Create DataFrame to hold TOI
    toi_in_shift = pd.DataFrame(index=pbp.index, columns=players)

    for pid in players:
        toi = 0.0
        on_shift = False
        toi_col = []

        for i in range(len(pbp)):
            # Check if player is currently on ice for this event
            player_on_ice = False
            for col in on_ice_cols:
                if pbp[col].iat[i] == pid:
                    player_on_ice = True
                    break
            
            # Handle shift changes
            if is_change.iat[i]:
                # Player going OFF - end their shift
                if pid in ids_off.iat[i].split(';'):
                    on_shift = False
                    toi = 0.0
                # Player coming ON - start their shift
                elif pid in ids_on.iat[i].split(';'):
                    on_shift = True
                    toi = 0.0
            else:
                # For non-change events, sync on_shift status with on-ice status
                # If player appears on ice but wasn't marked on_shift, start their shift
                if player_on_ice and not on_shift:
                    on_shift = True
                    toi = 0.0
                # If player is marked on_shift but not on ice, end their shift
                elif not player_on_ice and on_shift:
                    on_shift = False
                    toi = 0.0

            # Record TOI for this event (after determining shift status)
            if on_shift:
                toi += event_time.iat[i]
            
            toi_col.append(toi if on_shift else 0.0)

        toi_in_shift[pid] = toi_col

    #Attach on-ice TOI to on-ice columns
    for col in on_ice_cols:
        out = col.replace('_id', '_shift_time_on_ice')
        pbp[out] = np.nan

        pid_vals = pbp[col].values
        mask = ~pd.isna(pid_vals)
        pbp.loc[mask, out] = [
            toi_in_shift.at[i, pid] if pid in toi_in_shift.columns else np.nan
            for i, pid in zip(pbp.index[mask], pid_vals[mask])
        ]

    #Shooter
    shooter_col = 'event_player_1_shift_time_on_ice'
    pbp[shooter_col] = np.nan
    mask = pbp['event_player_1_id'].notna()
    pbp.loc[mask, shooter_col] = [
        toi_in_shift.at[i, pid] if pid in toi_in_shift.columns else np.nan
        for i, pid in zip(pbp.index[mask], pbp.loc[mask, 'event_player_1_id'])
    ]

    #Opponent TOI
    for i in range(1, 7):
        pbp[f'event_on_ice_against_{i}_shift_time_on_ice'] = np.where(
            pbp['event_team_venue'] == 'away',
            pbp[f'home_on_{i}_shift_time_on_ice'],
            pbp[f'away_on_{i}_shift_time_on_ice']
        )

    return pbp
        
def prep_xG_data(pbp):
    #Prep data for xG training and calculation
    pbp = fix_players(pbp)

    #Informal groupby
    pbp = pbp.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])

    #Recalibrate times series data with current data
    pbp['seconds_since_last'] = pbp['seconds_elapsed'] - pbp['seconds_elapsed'].shift(1) 
    #Prevent leaking between games by setting value to zero when no time has occured in game
    pbp["seconds_since_last"] = np.where(pbp['seconds_elapsed']==0,0,pbp['seconds_since_last'])

    #Create last event columns
    pbp["event_team_last"] = pbp['event_team_abbr'].shift(1)
    pbp["event_type_last"] = pbp['event_type'].shift(1)
    pbp["x_adj_last"] = pbp['x_adj'].shift(1)
    pbp["y_adj_last"] = pbp['y_adj'].shift(1)
    pbp["zone_code_last"] = pbp['zone_code'].shift(1)

    pbp = pbp.sort_values(['season','game_id','period','seconds_elapsed','event_num'])    

    #Contextual Data (for score state minimize the capture to four goals)
    pbp['score_state'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['away_score']-pbp['home_score'],pbp['home_score']-pbp['away_score'])
    pbp['score_state'] = np.where(pbp['score_state']>4,4,pbp['score_state'])
    pbp['score_state'] = np.where(pbp['score_state']<-4,-4,pbp['score_state'])

    pbp['strength_diff'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['away_skaters']-pbp['home_skaters'],pbp['home_skaters']-pbp['away_skaters'])
    pbp['strength_state_venue'] = pbp['away_skaters'].astype(str)+'v'+pbp['home_skaters'].astype(str)
    pbp['distance_from_last'] = np.sqrt((pbp['x_adj'] - pbp['x_adj_last'])**2 + (pbp['y_adj'] - pbp['y_adj_last'])**2)
    pbp['angle_from_last'] = np.degrees(np.arctan2(abs(pbp['y_adj'] - pbp['y_adj_last']), abs(89 - (pbp['x_adj']-pbp['x_adj_last']))))

    #Event speeds
    pbp['speed_from_last'] = np.where(pbp['seconds_since_last']==0,0,pbp['distance_from_last']/pbp['seconds_since_last'])
    pbp['speed_of_angle_from_last'] = np.where(pbp['seconds_since_last']==0,0,pbp['angle_from_last']/pbp['seconds_since_last'])

    #Rush, in-zone, and rebound shots are labelled
    pbp['rush'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['zone_code_last'].isin(['N','D']))&(pbp['zone_code']=='O')&(pbp['seconds_since_last']<=5),1,0)
    pbp['in_zone'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['zone_code_last']=='O')&(pbp['zone_code']=='O')&(pbp['seconds_since_last']<=5),1,0)
    pbp['rebound'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_type_last'].isin(wsba.FENWICK_EVENTS))&(pbp['seconds_since_last']<=2),1,0)

    #Create boolean variables
    pbp["is_goal"]=(pbp['event_type']=='goal').astype(int)
    pbp["is_home"]=(pbp['home_team_abbr']==pbp['event_team_abbr']).astype(int)

    #Boolean variables for shot types and prior events
    for shot in wsba.SHOT_TYPES:
        pbp[shot] = (pbp['shot_type']==shot).astype(int)
    for event in wsba.EVENTS[:-1]:
        pbp[f'prior_{event}'] = (pbp['event_type_last']==event).astype(int)

    pbp['other-shot'] = (~pbp['shot_type'].isin(wsba.SHOT_TYPES)).astype(int)
    pbp['prior_same'] = (pbp['event_team_last']==pbp['event_team_abbr']).astype(int)
    
    #Strength boolean (used instead of 'strength_diff' in order to more usefully distinguish between strength states)
    for strength in wsba.STRENGTHS:
        pbp[f'strength_{strength}'] = (pbp['strength_state']==strength).astype(int)
    
    #Flag special shot attempts added after 2021
    pbp['short'] = (pbp['event_reason']=='short').astype(int)
    pbp['failed_bank'] = (pbp['event_reason']=='failed-bank-attempt').astype(int)

    #Determine if the current play occured on the opposite side of the ice from the previous event
    pbp['cross_ice'] = (pbp['y_adj_last'] * pbp['y_adj'] < 0).astype(int)

    #Misc variables
    pbp['empty_net'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_goalie_id'].isna()),1,0)
    pbp['offwing'] = np.where(((pbp['y_adj']<0)&(pbp['event_player_1_hand']=='L'))|((pbp['y_adj']>=0)&(pbp['event_player_1_hand']=='R')),1,0)
    
    #Add shot assist probabilities
    pbp = apply_passing_imputation(pbp)

    #Add on-ice player time on ice
    #pbp = apply_time_on_ice(pbp)

    #Return: pbp data prepared to train and calculate the xG model
    return pbp

def wsba_xG(pbp, model_type: Literal['bayesian', 'frequentist'] = 'frequentist', states = False, hypertune = False, train = False, test_path = wsba.TEST_PATH, cv_path = wsba.CV_PATH, model_path = wsba.XG_MODEL, train_runs = 20, cv_runs = 20):
    #Train and calculate the WSBA Expected Goals model
    
    #Add index for future merging
    pbp['event_index'] = pbp.index

    #Initialize xG column for all events
    pbp['xG'] = 0.0

    #Recalibrate coordinates
    pbp = wsba.adjust_coords(pbp)

    #Recalculate stat states if speciifed
    if states:
        for venue in ['away','home']:
            pbp[f'{venue}_score'] = ((pbp['event_team_venue']==venue)&(pbp['event_type']=='goal')).groupby(pbp['game_id']).cumsum().shift(1)
            pbp[f'{venue}_corsi'] = ((pbp['event_team_venue']==venue)&(pbp['event_type'].isin(['blocked-shot','missed-shot','shot-on-goal','goal']))).groupby(pbp['game_id']).cumsum().shift(1)
            pbp[f'{venue}_fenwick'] = ((pbp['event_team_venue']==venue)&(pbp['event_type'].isin(['missed-shot','shot-on-goal','goal']))).groupby(pbp['game_id']).cumsum().shift(1)
            pbp[f'{venue}_penalties'] = ((pbp['event_team_venue']==venue)&(pbp['event_type']=='penalty')).groupby(pbp['game_id']).cumsum().shift(1)

    #Fix strengths
    pbp['strength_state'] = np.where((pbp['season_type']==3)&(pbp['period']>4),
                                    (np.where(pbp['event_team_abbr']==pbp['away_team_abbr'],
                                            pbp['away_skaters'].astype(str)+"v"+pbp['home_skaters'].astype(str),
                                            pbp['home_skaters'].astype(str)+"v"+pbp['away_skaters'].astype(str))),
                                    pbp['strength_state'])

    #Prep data and filter shot events
    data = prep_xG_data(pbp.loc[(pbp['event_type'].isin(wsba.EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())])
    data = data.loc[data['event_type'].isin(wsba.FENWICK_EVENTS)]
    
    if model_type == 'bayesian':
        NotImplementedError('PyMC Model in Development...')
    else:
        dfs = []
        for empty_net in [False, True]:
            #Two sub-models: Those on a goaltender and those on an empty net
            training = data.loc[data['empty_net']==1 if empty_net else data['empty_net']==0]

            #Calibrate paths
            if empty_net:
                test_path = test_path.replace('runs','runs_en')
                cv_path = cv_path.replace('runs','runs_en')
                model_path = model_path.replace('wsba_xg.json', 'wsba_xg_en.json')
            else:
                test_path = test_path
                cv_path = cv_path
                model_path = model_path

            #Convert to sparse
            data_sparse = sp.csr_matrix(training[[wsba.TARGET]+wsba.CONTINUOUS+wsba.BOOLEAN])
            is_goal_vect = data_sparse[:,0].toarray()
            predictors = data_sparse[:,1:]

            #XGB DataModel
            xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect,feature_names=(wsba.CONTINUOUS+wsba.BOOLEAN))

            if train:
                print('### XGBOOST MODEL TRAINING ###')
                if hypertune:
                    # Number of runs
                    run_num = train_runs

                    # DataFrames to store results
                    best_df = pd.DataFrame(columns=["max_depth", "eta", "gamma", "subsample", "colsample_bytree", "min_child_weight", "max_delta_step"])
                    best_ll = pd.DataFrame(columns=["ll", "ll_rounds", "auc", "auc_rounds", "seed"])

                    print('### HYPERTUNING ###')
                    # Loop
                    for i in range(run_num):
                        print(f"## LOOP: {i+1} ##")
                        
                        param = {
                            "objective": "binary:logistic",
                            "eval_metric": ["logloss", "auc"],
                            "max_depth": 6,
                            "eta": np.random.uniform(0.06, 0.11),
                            "gamma": np.random.uniform(0.06, 0.12),
                            "subsample": np.random.uniform(0.76, 0.84),
                            "colsample_bytree": np.random.uniform(0.76, 0.8),
                            "min_child_weight": np.random.randint(5, 23),
                            "max_delta_step": np.random.randint(4, 9)
                        }
                        
                        # Cross-validation
                        seed = np.random.randint(0, 10000)
                        np.random.seed(seed)
                        
                        cv_results = xgb.cv(
                            params=param,
                            dtrain=xgb_matrix,
                            num_boost_round=1000,
                            nfold=5,
                            early_stopping_rounds=25,
                            metrics=["logloss", "auc"],
                            seed=seed
                        )
                        
                        # Record results
                        best_df.loc[i] = param
                        best_ll.loc[i] = [
                            cv_results["test-logloss-mean"].min(),
                            cv_results["test-logloss-mean"].idxmin(),
                            cv_results["test-auc-mean"].max(),
                            cv_results["test-auc-mean"].idxmax(),
                            seed
                        ]

                    # Combine results
                    best_all = pd.concat([best_df, best_ll], axis=1).dropna()

                    # Arrange to get best run
                    best_all = best_all.sort_values(by="auc", ascending=False)

                    best_all.to_csv(test_path,index=False)

                    # Final parameters
                    param_7_EV = {
                        "objective": "binary:logistic",
                        "eval_metric": ["logloss", "auc"],
                        "gamma": best_all['gamma'].iloc[0],
                        "subsample": best_all['subsample'].iloc[0],
                        "max_depth": best_all['max_depth'].iloc[0],
                        "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                        "min_child_weight": best_all['min_child_weight'].iloc[0],
                        "max_delta_step": best_all['max_delta_step'].iloc[0],
                    }

                    # CV rounds Loop
                    run_num = cv_runs
                    cv_test = pd.DataFrame(columns=["AUC_rounds", "AUC", "LL_rounds", "LL", "seed"])

                    print('### CROSS-VALIDATION ###')
                    for i in range(run_num):
                        print(f"## LOOP: {i+1} ##")
                        
                        seed = np.random.randint(0, 10000)
                        np.random.seed(seed)
                        
                        cv_rounds = xgb.cv(
                            params=param_7_EV,
                            dtrain=xgb_matrix,
                            num_boost_round=1000,
                            nfold=5,
                            early_stopping_rounds=25,
                            metrics=["logloss", "auc"],
                            seed=seed
                        )
                        
                        # Record results
                        cv_test.loc[i] = [
                            cv_rounds["test-auc-mean"].idxmax(),
                            cv_rounds["test-auc-mean"].max(),
                            cv_rounds["test-logloss-mean"].idxmin(),
                            cv_rounds["test-logloss-mean"].min(),
                            seed
                        ]

                    # Clean results and sort to find the number of rounds to use and seed
                    cv_final = cv_test.sort_values(by="AUC", ascending=False)
                    cv_final.to_csv(cv_path,index=False)
                else:
                    # Load previous parameters
                    best_all = pd.read_csv(test_path)
                    cv_final = pd.read_csv(cv_path)

                    print('Loaded hyperparameters...')
                    # Final parameters
                    param_7_EV = {
                        "objective": "binary:logistic",
                        "eval_metric": ["logloss", "auc"],
                        "gamma": best_all['gamma'].iloc[0],
                        "subsample": best_all['subsample'].iloc[0],
                        "max_depth": best_all['max_depth'].iloc[0],
                        "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                        "min_child_weight": best_all['min_child_weight'].iloc[0],
                        "max_delta_step": best_all['max_delta_step'].iloc[0],
                    }

                print('Training model...')
                seed = int(cv_final['seed'].iloc[0])
                np.random.seed(seed)
                model = xgb.train(
                    params=param_7_EV,
                    dtrain=xgb_matrix,
                    num_boost_round=int(cv_final['AUC_rounds'].iloc[0]),
                    verbose_eval=2,
                )
                
                #Save model
                model.save_model(model_path)
                
            else:
                #Load model
                model = xgb.Booster()
                model.load_model(model_path)

                if len(training) > 0:
                    #Predict xG for fenwick shots
                    training['xG'] = model.predict(xgb_matrix)

                dfs.append(training)

        if not train:
            xg_data = pd.concat(dfs)

            #Add xG columns
            for col in xg_data.columns:
                if col not in pbp.columns:
                    pbp[col] = np.nan
                    pbp[col] = pbp[col].astype('object')

            pbp.loc[xg_data.index, xg_data.columns] = xg_data

            #Return: PBP dataframe with xG columns
            pbp_xg = pbp.sort_values(by=['game_id','period','seconds_elapsed','event_num'])

            return pbp_xg

def feature_importance(model_path = wsba.XG_MODEL, metric_path = wsba.METRIC_PATH):
    print('Feature importance for WSBA xG Model...')
    
    for en in [False, True]:
        model = xgb.Booster()
        model.load_model(model_path if not en else model_path.replace('.json', '_en.json'))

        fi = pd.DataFrame(model.get_score(importance_type='gain').items(), columns=['feature','gain']).sort_values("gain", ascending=False)
        fi['gain'] /= fi['gain'].sum()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(fi['feature'], fi['gain'])
        ax.invert_yaxis()

        ax.set_xlabel('Gain')
        ax.set_title(f'WSBA xG {'(Empty Net) ' if en else ''}Feature Importance')

        plt.savefig(os.path.join(metric_path,f'feature_importance{'_en' if en else ''}.png'),bbox_inches='tight')

def roc_auc_curve(pbp, metric_path = wsba.METRIC_PATH):
    print('ROC-AUC Curve for WSBA xG Model...')

    if 'xG' in pbp.columns:
        data = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())]
          
        fpr, tpr, _ = roc_curve(data['is_goal'], data['xG'])
        roc_auc = auc(fpr,tpr)
        
        plt.figure()
        plt.plot(fpr,tpr,label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"WSBA xG ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(metric_path, f'roc_auc_curve.png'), bbox_inches='tight')
    else:
        print('No xG found for provided play-by-play data.  Apply xG model to the play-by-play data first.')

def calibration(pbp, metric_path = wsba.METRIC_PATH):
    print('Reliability for WSBA xG Model...')

    if 'xG' in pbp.columns: 
        data = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())]
        fop, mpv = calibration_curve(data['is_goal'], data['xG'], strategy='uniform')

        plt.figure()
        plt.plot(mpv, fop, "s-", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.title(f"WSBA xG Calibration")
        plt.xlabel("Predicted Probability (mean)")
        plt.ylabel("Fraction of positives")
        plt.legend(loc="best")
        plt.savefig(os.path.join(metric_path, f'calibration.png'), bbox_inches='tight')

    else:
        print('No xG found for provided play-by-play data.  Apply xG model to the play-by-play data first.')
