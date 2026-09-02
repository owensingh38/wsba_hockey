import os
import json as json_lib

import polars as pl

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


def _indicator(expr):
    """Cast nullable boolean expressions to stable integer indicators."""
    return expr.fill_null(False).cast(pl.Int64)


def _xg_model_features(model, model_path: str) -> list[str]:
    feature_path = model_path.replace('.json', '_features.json')
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            return list(json_lib.load(f))
    if model.feature_names:
        return list(model.feature_names)
    return CONTINUOUS + BOOLEAN


def _prep_xg_data(pbp: pl.DataFrame) -> pl.DataFrame:
    pbp = fix_players(pbp)
    order = ['season', 'game_id', 'period', 'seconds_elapsed', 'event_num']
    pbp = pbp.sort(order)
    pbp = pbp.with_columns([
        (pl.col('seconds_elapsed') - pl.col('seconds_elapsed').shift(1)).alias('seconds_since_last'),
        pl.col('event_team_abbr').shift(1).alias('event_team_last'),
        pl.col('event_type').shift(1).alias('event_type_last'),
        pl.col('x_adj').shift(1).alias('x_adj_last'),
        pl.col('y_adj').shift(1).alias('y_adj_last'),
        pl.col('zone_code').shift(1).alias('zone_code_last'),
    ]).with_columns([
        pl.when(pl.col('seconds_elapsed') == 0).then(0).otherwise(pl.col('seconds_since_last')).alias('seconds_since_last'),
        pl.when(pl.col('away_team_abbr') == pl.col('event_team_abbr')).then(pl.col('away_score') - pl.col('home_score')).otherwise(pl.col('home_score') - pl.col('away_score')).clip(-4, 4).alias('score_state'),
        pl.when(pl.col('away_team_abbr') == pl.col('event_team_abbr')).then(pl.col('away_skaters') - pl.col('home_skaters')).otherwise(pl.col('home_skaters') - pl.col('away_skaters')).alias('strength_diff'),
        (pl.col('away_skaters').cast(pl.String) + 'v' + pl.col('home_skaters').cast(pl.String)).alias('strength_state_venue'),
        ((pl.col('x_adj') - pl.col('x_adj_last')).pow(2) + (pl.col('y_adj') - pl.col('y_adj_last')).pow(2)).sqrt().alias('distance_from_last'),
        pl.arctan2((pl.col('y_adj') - pl.col('y_adj_last')).abs(), (89 - (pl.col('x_adj') - pl.col('x_adj_last'))).abs()).degrees().alias('angle_from_last'),
    ]).with_columns([
        pl.when(pl.col('seconds_since_last') == 0).then(0).otherwise(pl.col('distance_from_last') / pl.col('seconds_since_last')).alias('speed_from_last'),
        pl.when(pl.col('seconds_since_last') == 0).then(0).otherwise(pl.col('angle_from_last') / pl.col('seconds_since_last')).alias('speed_of_angle_from_last'),
        _indicator(pl.col('event_type').is_in(FENWICK_EVENTS) & pl.col('zone_code_last').is_in(['N', 'D']) & (pl.col('zone_code') == 'O') & (pl.col('seconds_since_last') <= 5)).alias('rush'),
        _indicator(pl.col('event_type').is_in(FENWICK_EVENTS) & (pl.col('zone_code_last') == 'O') & (pl.col('zone_code') == 'O') & (pl.col('seconds_since_last') <= 5)).alias('in_zone'),
        _indicator(pl.col('event_type').is_in(FENWICK_EVENTS) & pl.col('event_type_last').is_in(FENWICK_EVENTS) & (pl.col('seconds_since_last') <= 2)).alias('rebound'),
        _indicator(pl.col('event_type') == 'goal').alias('is_goal'),
        _indicator(pl.col('home_team_abbr') == pl.col('event_team_abbr')).alias('is_home'),
    ])

    for shot in SHOT_TYPES:
        pbp = pbp.with_columns(_indicator(pl.col('shot_type') == shot).alias(shot))
    for event in EVENTS[:-1]:
        pbp = pbp.with_columns(_indicator(pl.col('event_type_last') == event).alias(f'prior_{event}'))

    pbp = pbp.with_columns([
        _indicator(~pl.col('shot_type').is_in(SHOT_TYPES)).alias('other-shot'),
        _indicator(pl.col('event_team_last') == pl.col('event_team_abbr')).alias('prior_same'),
    ])

    for strength in STRENGTHS:
        pbp = pbp.with_columns(_indicator(pl.col('strength_state') == strength).alias(f'strength_{strength}'))

    pbp = pbp.with_columns([
        _indicator(pl.col('event_reason') == 'short').alias('short'),
        _indicator(pl.col('event_reason') == 'failed-bank-attempt').alias('failed_bank'),
        _indicator(pl.col('y_adj_last') * pl.col('y_adj') < 0).alias('cross_ice'),
        _indicator(pl.col('event_type').is_in(FENWICK_EVENTS) & pl.col('event_goalie_id').is_null()).alias('empty_net'),
        _indicator(((pl.col('y_adj') < 0) & (pl.col('event_player_1_hand') == 'L')) | ((pl.col('y_adj') >= 0) & (pl.col('event_player_1_hand') == 'R'))).alias('offwing'),
    ])

    return apply_passing_imputation(pbp)


def _recalculate_xg_states(pbp: pl.DataFrame) -> pl.DataFrame:
    pbp = pbp.clone()
    for venue in ['away', 'home']:
        for name, events in {
            'score': ['goal'],
            'corsi': ['blocked-shot', 'missed-shot', 'shot-on-goal', 'goal'],
            'fenwick': ['missed-shot', 'shot-on-goal', 'goal'],
            'penalties': ['penalty'],
        }.items():
            pbp = pbp.with_columns(
                ((pl.col('event_team_venue') == venue) & pl.col('event_type').is_in(events)).cast(pl.Int64).cum_sum().over('game_id').shift(1).fill_null(0).alias(f'{venue}_{name}')
            )
    return pbp


def _apply_xg_model(pbp: pl.DataFrame, model_path: str, states: bool = False) -> pl.DataFrame:
    import scipy.sparse as sp
    import xgboost as xgb

    pbp = pbp.drop('event_index', strict=False).clone().with_columns(pl.int_range(0, pl.len()).alias('event_index'))
    pbp = pbp.with_columns(pl.lit(0.0).alias('xG'))
    pbp = adjust_coords(pbp)

    if states:
        pbp = _recalculate_xg_states(pbp)

    pbp = pbp.with_columns(
        pl.when((pl.col('season_type') == 3) & (pl.col('period') > 4))
        .then(pl.when(pl.col('event_team_abbr') == pl.col('away_team_abbr')).then(pl.col('away_skaters').cast(pl.String) + 'v' + pl.col('home_skaters').cast(pl.String)).otherwise(pl.col('home_skaters').cast(pl.String) + 'v' + pl.col('away_skaters').cast(pl.String)))
        .otherwise(pl.col('strength_state')).alias('strength_state')
    )

    data = _prep_xg_data(
        pbp.filter(
            pl.col('event_type').is_in(EVENTS) &
            pl.col('strength_state').is_in(STRENGTHS) &
            pl.col('x').is_not_null() &
            pl.col('y').is_not_null()
        )
    )
    data = data.filter(pl.col('event_type').is_in(FENWICK_EVENTS))
    if data.is_empty():
        return pbp.sort(['game_id', 'period', 'seconds_elapsed', 'event_num'])

    dfs = []
    for empty_net in [False, True]:
        training = data.filter(pl.col('empty_net') == (1 if empty_net else 0))
        if training.is_empty():
            continue

        current_model_path = model_path.replace('wsba_xg.json', 'wsba_xg_en.json') if empty_net else model_path
        model = xgb.Booster()
        model.load_model(current_model_path)
        features = _xg_model_features(model, current_model_path)
        for feature in features:
            if feature not in training.columns:
                training = training.with_columns(pl.lit(0.0).alias(feature))

        data_sparse = sp.csr_matrix(training[[TARGET] + features])
        is_goal_vect = data_sparse[:, 0].toarray()
        predictors = data_sparse[:, 1:]
        xgb_matrix = xgb.DMatrix(
            data=predictors,
            label=is_goal_vect,
            feature_names=features
        )
        training = training.with_columns(pl.Series('xG', model.predict(xgb_matrix)))
        dfs.append(training)

    if dfs:
        xg_data = pl.concat(dfs, how='diagonal_relaxed')
        new_cols = [col for col in xg_data.columns if col not in {'event_index'}]
        updates = xg_data.select(['event_index'] + [pl.col(col).alias(f'{col}_new') for col in new_cols])
        pbp = pbp.join(updates, on='event_index', how='left')
        pbp = pbp.with_columns([
            pl.coalesce([pl.col(f'{col}_new'), pl.col(col)]).alias(col) if col in pbp.columns else pl.col(f'{col}_new').alias(col)
            for col in new_cols
        ]).drop([f'{col}_new' for col in new_cols])

    return pbp.sort(['game_id', 'period', 'seconds_elapsed', 'event_num'])


def nhl_apply_xG(pbp: pl.DataFrame, states: bool = False) -> pl.DataFrame:
    """
    Given play-by-play data, return this data with xG-related columns.

    Args:
        pbp (pl.DataFrame):
            A DataFrame containing play-by-play data generated within the WBSA Hockey package.
        states (bool, optional):
            If True, recalculate score, Corsi, Fenwick, and penalty states before applying xG.
            Defaults to False, matching the previous xG module behavior.

    Returns:
        pl.DataFrame:
            A DataFrame containing input play-by-play data with xG column.
    """
    print(f"Applying WSBA xG to model with seasons: {pbp['season'].unique().to_list()}")

    legacy = pbp.filter(pl.col('season') < 20232024)
    modern = pbp.filter(pl.col('season') >= 20232024)

    model_dir = os.path.dirname(XG_MODEL)
    model_jobs = [
        (legacy, os.path.join(model_dir, 'legacy', os.path.basename(XG_MODEL))),
        (modern, XG_MODEL),
    ]
    return pl.concat([
        _apply_xg_model(df, model_path, states=states)
        for df, model_path in model_jobs
        if not df.is_empty()
    ])
