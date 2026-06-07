import os
import json as json_lib

import numpy as np
import pandas as pd

from wsba_hockey.tools.scraping import adjust_coords, apply_passing_imputation, fix_players
from wsba_hockey.tools.globals import (
    BOOLEAN,
    CONTINUOUS,
    EVENTS,
    FENWICK_EVENTS,
    SHOT_TYPES,
    STRENGTHS,
    TARGET,
    XG_MODEL,
)


def _xg_model_features(model, model_path: str) -> list[str]:
    feature_path = model_path.replace('.json', '_features.json')
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            return list(json_lib.load(f))
    if model.feature_names:
        return list(model.feature_names)
    return CONTINUOUS + BOOLEAN


def _prep_xg_data(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = fix_players(pbp)
    pbp = pbp.sort_values(by=['season', 'game_id', 'period', 'seconds_elapsed', 'event_num'])

    pbp['seconds_since_last'] = pbp['seconds_elapsed'] - pbp['seconds_elapsed'].shift(1)
    pbp['seconds_since_last'] = np.where(pbp['seconds_elapsed'] == 0, 0, pbp['seconds_since_last'])

    pbp['event_team_last'] = pbp['event_team_abbr'].shift(1)
    pbp['event_type_last'] = pbp['event_type'].shift(1)
    pbp['x_adj_last'] = pbp['x_adj'].shift(1)
    pbp['y_adj_last'] = pbp['y_adj'].shift(1)
    pbp['zone_code_last'] = pbp['zone_code'].shift(1)

    pbp = pbp.sort_values(['season', 'game_id', 'period', 'seconds_elapsed', 'event_num'])

    pbp['score_state'] = np.where(
        pbp['away_team_abbr'] == pbp['event_team_abbr'],
        pbp['away_score'] - pbp['home_score'],
        pbp['home_score'] - pbp['away_score']
    )
    pbp['score_state'] = np.clip(pbp['score_state'], -4, 4)
    pbp['strength_diff'] = np.where(
        pbp['away_team_abbr'] == pbp['event_team_abbr'],
        pbp['away_skaters'] - pbp['home_skaters'],
        pbp['home_skaters'] - pbp['away_skaters']
    )
    pbp['strength_state_venue'] = pbp['away_skaters'].astype(str) + 'v' + pbp['home_skaters'].astype(str)
    pbp['distance_from_last'] = np.sqrt((pbp['x_adj'] - pbp['x_adj_last'])**2 + (pbp['y_adj'] - pbp['y_adj_last'])**2)
    pbp['angle_from_last'] = np.degrees(
        np.arctan2(abs(pbp['y_adj'] - pbp['y_adj_last']), abs(89 - (pbp['x_adj'] - pbp['x_adj_last'])))
    )
    pbp['speed_from_last'] = np.where(
        pbp['seconds_since_last'] == 0,
        0,
        pbp['distance_from_last'] / pbp['seconds_since_last']
    )
    pbp['speed_of_angle_from_last'] = np.where(
        pbp['seconds_since_last'] == 0,
        0,
        pbp['angle_from_last'] / pbp['seconds_since_last']
    )
    pbp['rush'] = np.where(
        (pbp['event_type'].isin(FENWICK_EVENTS)) &
        (pbp['zone_code_last'].isin(['N', 'D'])) &
        (pbp['zone_code'] == 'O') &
        (pbp['seconds_since_last'] <= 5),
        1,
        0
    )
    pbp['in_zone'] = np.where(
        (pbp['event_type'].isin(FENWICK_EVENTS)) &
        (pbp['zone_code_last'] == 'O') &
        (pbp['zone_code'] == 'O') &
        (pbp['seconds_since_last'] <= 5),
        1,
        0
    )
    pbp['rebound'] = np.where(
        (pbp['event_type'].isin(FENWICK_EVENTS)) &
        (pbp['event_type_last'].isin(FENWICK_EVENTS)) &
        (pbp['seconds_since_last'] <= 2),
        1,
        0
    )
    pbp['is_goal'] = (pbp['event_type'] == 'goal').astype(int)
    pbp['is_home'] = (pbp['home_team_abbr'] == pbp['event_team_abbr']).astype(int)

    for shot in SHOT_TYPES:
        pbp[shot] = (pbp['shot_type'] == shot).astype(int)
    for event in EVENTS[:-1]:
        pbp[f'prior_{event}'] = (pbp['event_type_last'] == event).astype(int)

    pbp['other-shot'] = (~pbp['shot_type'].isin(SHOT_TYPES)).astype(int)
    pbp['prior_same'] = (pbp['event_team_last'] == pbp['event_team_abbr']).astype(int)

    for strength in STRENGTHS:
        pbp[f'strength_{strength}'] = (pbp['strength_state'] == strength).astype(int)

    pbp['short'] = (pbp['event_reason'] == 'short').astype(int)
    pbp['failed_bank'] = (pbp['event_reason'] == 'failed-bank-attempt').astype(int)
    pbp['cross_ice'] = (pbp['y_adj_last'] * pbp['y_adj'] < 0).astype(int)
    pbp['empty_net'] = np.where(
        (pbp['event_type'].isin(FENWICK_EVENTS)) & (pbp['event_goalie_id'].isna()),
        1,
        0
    )
    pbp['offwing'] = np.where(
        ((pbp['y_adj'] < 0) & (pbp['event_player_1_hand'] == 'L')) |
        ((pbp['y_adj'] >= 0) & (pbp['event_player_1_hand'] == 'R')),
        1,
        0
    )

    return apply_passing_imputation(pbp)


def _recalculate_xg_states(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    for venue in ['away', 'home']:
        pbp[f'{venue}_score'] = (
            (pbp['event_team_venue'] == venue) & (pbp['event_type'] == 'goal')
        ).groupby(pbp['game_id']).cumsum().shift(1)
        pbp[f'{venue}_corsi'] = (
            (pbp['event_team_venue'] == venue) &
            (pbp['event_type'].isin(['blocked-shot', 'missed-shot', 'shot-on-goal', 'goal']))
        ).groupby(pbp['game_id']).cumsum().shift(1)
        pbp[f'{venue}_fenwick'] = (
            (pbp['event_team_venue'] == venue) &
            (pbp['event_type'].isin(['missed-shot', 'shot-on-goal', 'goal']))
        ).groupby(pbp['game_id']).cumsum().shift(1)
        pbp[f'{venue}_penalties'] = (
            (pbp['event_team_venue'] == venue) & (pbp['event_type'] == 'penalty')
        ).groupby(pbp['game_id']).cumsum().shift(1)
    return pbp


def _apply_xg_model(pbp: pd.DataFrame, model_path: str, states: bool = False) -> pd.DataFrame:
    import scipy.sparse as sp
    import xgboost as xgb

    pbp = pbp.copy()
    pbp['event_index'] = pbp.index
    pbp['xG'] = 0.0
    pbp = adjust_coords(pbp)

    if states:
        pbp = _recalculate_xg_states(pbp)

    pbp['strength_state'] = np.where(
        (pbp['season_type'] == 3) & (pbp['period'] > 4),
        np.where(
            pbp['event_team_abbr'] == pbp['away_team_abbr'],
            pbp['away_skaters'].astype(str) + "v" + pbp['home_skaters'].astype(str),
            pbp['home_skaters'].astype(str) + "v" + pbp['away_skaters'].astype(str)
        ),
        pbp['strength_state']
    )

    data = _prep_xg_data(
        pbp.loc[
            (pbp['event_type'].isin(EVENTS)) &
            (pbp['strength_state'].isin(STRENGTHS)) &
            (pbp['x'].notna()) &
            (pbp['y'].notna())
        ].copy()
    )
    data = data.loc[data['event_type'].isin(FENWICK_EVENTS)].copy()
    if data.empty:
        return pbp.sort_values(by=['game_id', 'period', 'seconds_elapsed', 'event_num'])

    dfs = []
    for empty_net in [False, True]:
        training = data.loc[data['empty_net'].eq(1 if empty_net else 0)].copy()
        if training.empty:
            continue

        current_model_path = model_path.replace('wsba_xg.json', 'wsba_xg_en.json') if empty_net else model_path
        model = xgb.Booster()
        model.load_model(current_model_path)
        features = _xg_model_features(model, current_model_path)
        for feature in features:
            if feature not in training.columns:
                training[feature] = 0.0

        data_sparse = sp.csr_matrix(training[[TARGET] + features])
        is_goal_vect = data_sparse[:, 0].toarray()
        predictors = data_sparse[:, 1:]
        xgb_matrix = xgb.DMatrix(
            data=predictors,
            label=is_goal_vect,
            feature_names=features
        )
        training['xG'] = model.predict(xgb_matrix)
        dfs.append(training)

    if dfs:
        xg_data = pd.concat(dfs)
        for col in xg_data.columns:
            if col not in pbp.columns:
                pbp[col] = np.nan
                pbp[col] = pbp[col].astype('object')
        pbp.loc[xg_data.index, xg_data.columns] = xg_data

    return pbp.sort_values(by=['game_id', 'period', 'seconds_elapsed', 'event_num'])


def nhl_apply_xG(pbp: pd.DataFrame, states: bool = False) -> pd.DataFrame:
    """
    Given play-by-play data, return this data with xG-related columns.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play data generated within the WBSA Hockey package.
        states (bool, optional):
            If True, recalculate score, Corsi, Fenwick, and penalty states before applying xG.
            Defaults to False, matching the previous xG module behavior.

    Returns:
        pd.DataFrame:
            A DataFrame containing input play-by-play data with xG column.
    """
    print(f"Applying WSBA xG to model with seasons: {pbp['season'].drop_duplicates().to_list()}")

    legacy = pbp.loc[pbp['season'] < 20232024]
    modern = pbp.loc[pbp['season'] >= 20232024]

    model_dir = os.path.dirname(XG_MODEL)
    model_jobs = [
        (legacy, os.path.join(model_dir, 'legacy', os.path.basename(XG_MODEL))),
        (modern, XG_MODEL),
    ]
    return pd.concat([
        _apply_xg_model(df, model_path, states=states)
        for df, model_path in model_jobs
        if not df.empty
    ])
