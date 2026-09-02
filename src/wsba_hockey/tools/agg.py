from functools import lru_cache

import polars as pl

from wsba_hockey.tools.globals import (
    AGG_POST_METRICS, BIO_STAT_COL, FENWICK_EVENTS, NON_FINAL_STATES,
    NON_TOTALS, OPS, SHOT_TYPES, SPECIAL_KEYS,
)

@lru_cache(maxsize=8)
def _read_csv_cached(path):
    return pl.read_csv(path)


def _flag(expr):
    return expr.fill_null(False).cast(pl.Int64)


def _aggregate(df, keys, mapping):
    return df.group_by(keys, maintain_order=True).agg([
        (pl.col(source).n_unique() if op == 'nunique' else pl.col(source).sum()).alias(name)
        for name, (source, op) in mapping.items()
    ])


def process_stats(df, group, venue, game_strength, second_group):
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
    keep = ['season', 'game_id', 'strength_state', 'event_num', team_col, opp_col,
            'event_type', 'event_team_venue', 'event_team_abbr', 'ids_on', 'shift_type',
            'event_length', 'zone_code', 'penalty_duration', 'penalty_type', 'short', 'xG']
    if group == 'skater':
        keep += id_cols + primary_cols + secondary_cols + tertiary_cols
    elif group == 'goalie':
        keep += id_cols
    df = df.select(keep)
    if game_strength != 'all':
        df = df.with_columns(pl.when(pl.col('event_team_abbr') == pl.col(team_col)).then(pl.col('strength_state')).otherwise(pl.col('strength_state').cast(pl.String).str.reverse()).alias('strength_state_for')).filter(pl.col('strength_state_for').is_in(game_strength))
    if group == 'skater':
        # Turn the six on-ice slots into list columns and explode once.  This
        # avoids six frame copies and six concatenation inputs.
        stacked = [
            'player_id',
            'primary_fenwick_assist_probability',
            'secondary_fenwick_assist_probability',
            'tertiary_fenwick_assist_probability',
        ]
        df = df.with_columns([
            pl.concat_list([pl.col(col) for col in id_cols]).alias(stacked[0]),
            pl.concat_list([pl.col(col) for col in primary_cols]).alias(stacked[1]),
            pl.concat_list([pl.col(col) for col in secondary_cols]).alias(stacked[2]),
            pl.concat_list([pl.col(col) for col in tertiary_cols]).alias(stacked[3]),
        ]).drop(id_cols + primary_cols + secondary_cols + tertiary_cols).explode(stacked).filter(pl.col('player_id') > 0)
    elif group == 'goalie':
        df = df.rename({id_cols[0]: 'player_id'})

    event, zone, duration = pl.col('event_type'), pl.col('zone_code'), pl.col('penalty_duration')
    team = pl.col('event_team_abbr') == pl.col(team_col)
    opponent = pl.col('event_team_abbr') == pl.col(opp_col)
    valid = pl.col('short') == 0
    df = df.with_columns([
        pl.when(pl.col('short') == 1).then(0).otherwise(pl.col('xG')).alias('xG'),
        pl.when(team & valid).then(pl.col('xG')).otherwise(0.0).alias('expected_goals_for'),
        pl.when(opponent & valid).then(pl.col('xG')).otherwise(0.0).alias('expected_goals_against'),
        _flag((event == 'goal') & team).alias('goals_for'), _flag((event == 'goal') & opponent).alias('goals_against'),
        _flag(event.is_in(['shot-on-goal', 'goal']) & team).alias('shots_for'), _flag(event.is_in(['shot-on-goal', 'goal']) & opponent).alias('shots_against'),
        _flag(event.is_in(FENWICK_EVENTS) & valid & team).alias('fenwick_for'), _flag(event.is_in(FENWICK_EVENTS) & valid & opponent).alias('fenwick_against'),
        _flag(event.is_in(FENWICK_EVENTS + ['blocked-shot']) & team).alias('corsi_for'), _flag(event.is_in(FENWICK_EVENTS + ['blocked-shot']) & opponent).alias('corsi_against'),
        _flag((event == 'faceoff') & (((zone == 'O') & team) | ((zone == 'D') & opponent))).alias('offensive_zone_faceoffs'),
        _flag((event == 'faceoff') & (zone == 'N')).alias('neutral_zone_faceoffs'),
        _flag((event == 'faceoff') & (((zone == 'D') & team) | ((zone == 'O') & opponent))).alias('defensive_zone_faceoffs'),
    ])
    if group == 'skater':
        df = df.with_columns([
            pl.when(team & valid).then(pl.col('primary_fenwick_assist_probability')).otherwise(0.0).alias('primary_fenwick_assists'),
            pl.when(team & valid).then(pl.col('secondary_fenwick_assist_probability')).otherwise(0.0).alias('secondary_fenwick_assists'),
            pl.when(team & valid).then(pl.col('tertiary_fenwick_assist_probability')).otherwise(0.0).alias('tertiary_fenwick_assists'),
        ]).with_columns([
            pl.when(team & valid).then(pl.col('primary_fenwick_assist_probability') * pl.col('xG')).otherwise(0.0).alias('primary_expected_assists'),
            pl.when(team & valid).then(pl.col('secondary_fenwick_assist_probability') * pl.col('xG')).otherwise(0.0).alias('secondary_expected_assists'),
        ])
    elif group == 'team':
        df = df.with_columns([
            _flag((event == 'hit') & team).alias('hits_for'), _flag((event == 'hit') & opponent).alias('hits_against'),
            _flag((event == 'penalty') & team).alias('penalties_for'), _flag((event == 'penalty') & (duration == 2) & team).alias('minor_penalties_for'), _flag((event == 'penalty') & (duration == 5) & team).alias('major_penalties_for'), _flag((event == 'penalty') & (pl.col('penalty_type') == 'fighting') & team).alias('fighting_penalties_for'), pl.when((event == 'penalty') & team).then(duration).otherwise(0).alias('penalty_minutes_for'),
            _flag((event == 'penalty') & opponent).alias('penalties_against'), _flag((event == 'penalty') & (duration == 2) & opponent).alias('minor_penalties_against'), _flag((event == 'penalty') & (duration == 5) & opponent).alias('major_penalties_against'), _flag((event == 'penalty') & (pl.col('penalty_type') == 'fighting') & opponent).alias('fighting_penalties_against'), pl.when((event == 'penalty') & opponent).then(duration).otherwise(0).alias('penalty_minutes_against'),
            _flag((event == 'giveaway') & team).alias('giveaways'), _flag((event == 'takeaway') & team).alias('takeaways'),
        ]).with_columns((pl.col('corsi_against') - pl.col('fenwick_against')).alias('blocked_shots'))
    keys = [team_col] + (['player_id'] if group in ('skater', 'goalie') else []) + list(second_group)
    mapping = {'games_played': ('game_id', 'nunique'), 'time_on_ice': ('event_length', 'sum'), **{c: (c, 'sum') for c in ['fenwick_for', 'fenwick_against', 'goals_for', 'goals_against', 'shots_for', 'shots_against', 'expected_goals_for', 'expected_goals_against', 'corsi_for', 'corsi_against', 'offensive_zone_faceoffs', 'neutral_zone_faceoffs', 'defensive_zone_faceoffs']}}
    if group == 'skater':
        mapping.update({c: (c, 'sum') for c in ['primary_fenwick_assists', 'secondary_fenwick_assists', 'tertiary_fenwick_assists', 'primary_expected_assists', 'secondary_expected_assists']})
    if group == 'team':
        mapping.update({c: (c, 'sum') for c in ['hits_for', 'hits_against', 'penalties_for', 'minor_penalties_for', 'major_penalties_for', 'fighting_penalties_for', 'penalty_minutes_for', 'penalties_against', 'minor_penalties_against', 'major_penalties_against', 'fighting_penalties_against', 'penalty_minutes_against', 'giveaways', 'takeaways', 'blocked_shots']})
    stats = _aggregate(df, keys, mapping).rename({team_col: 'team_abbr'})
    return stats.drop([c for c in stats.columns if '_fenwick_assist_probability' in c], strict=False)


def calc_indv(pbp, game_strength, second_group):
    if game_strength != 'all':
        pbp = pbp.filter(pl.col('strength_state').is_in(game_strength))
    pbp = pbp.with_columns([
        pl.when(pl.col('event_team_abbr').is_not_null()).then(pl.when(pl.col('event_team_abbr') == pl.col('home_team_abbr')).then(pl.col('away_team_abbr')).otherwise(pl.col('home_team_abbr'))).otherwise(None).alias('event_team_abbr_2'),
        _flag(pl.col('event_type') == 'goal').alias('is_goal'), _flag(pl.col('event_type').is_in(['shot-on-goal', 'goal'])).alias('is_shot'), _flag(pl.col('event_type').is_in(FENWICK_EVENTS)).alias('is_fenwick'), _flag(pl.col('event_type').is_in(FENWICK_EVENTS + ['blocked-shot'])).alias('is_corsi'), _flag((pl.col('event_type') == 'blocked-shot') & (pl.col('event_reason') != 'teammate-blocked').fill_null(True)).alias('is_block'), _flag(pl.col('event_type') == 'hit').alias('is_hit'), _flag(pl.col('event_type') == 'giveaway').alias('is_giveaway'), _flag(pl.col('event_type') == 'takeaway').alias('is_takeaway'), _flag(pl.col('event_type') == 'penalty').alias('is_penalty'), _flag(pl.col('event_type') == 'faceoff').alias('is_faceoff'), _flag(pl.col('penalty_duration') == 2).alias('is_minor'), _flag(pl.col('penalty_duration') == 5).alias('is_major'), _flag(pl.col('penalty_type') == 'fighting').alias('is_fighting'),
    ]).with_columns([pl.when(pl.col('event_type') == 'goal').then(pl.col('event_team_abbr')).otherwise(pl.col('event_team_abbr_2')).alias('event_team_abbr_2'), pl.when(pl.col('short') == 1).then(0).otherwise(pl.col('is_fenwick')).alias('is_fenwick'), pl.when(pl.col('short') == 1).then(0).otherwise(pl.col('xG')).alias('xG')])
    events = pbp.filter(pl.col('event_type').is_in(['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot', 'hit', 'giveaway', 'takeaway', 'faceoff', 'penalty']))
    first = {'goals': ('is_goal', 'sum'), 'shots': ('is_shot', 'sum'), 'fenwick': ('is_fenwick', 'sum'), 'corsi': ('is_corsi', 'sum'), 'expected_goals': ('xG', 'sum'), 'hits_applied': ('is_hit', 'sum'), 'giveaways': ('is_giveaway', 'sum'), 'takeaways': ('is_takeaway', 'sum'), 'penalties_taken': ('is_penalty', 'sum'), 'minor_penalties_taken': ('is_minor', 'sum'), 'major_penalties_taken': ('is_major', 'sum'), 'fighting_penalties_taken': ('is_fighting', 'sum'), 'penalty_minutes_taken': ('penalty_duration', 'sum'), 'faceoff_wins': ('is_faceoff', 'sum')}
    second = {'primary_assists': ('is_goal', 'sum'), 'hits_received': ('is_hit', 'sum'), 'penalties_drawn': ('is_penalty', 'sum'), 'minor_penalties_drawn': ('is_minor', 'sum'), 'major_penalties_drawn': ('is_major', 'sum'), 'fighting_penalties_drawn': ('is_fighting', 'sum'), 'penalty_minutes_drawn': ('penalty_duration', 'sum'), 'faceoff_losses': ('is_faceoff', 'sum'), 'blocked_shots': ('is_block', 'sum')}
    clean = ['player_id', 'team_abbr'] + list(second_group)
    ep1 = _aggregate(events, ['event_player_1_id', 'event_team_abbr'] + list(second_group), first).rename({'event_player_1_id': 'player_id', 'event_team_abbr': 'team_abbr'}).with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    ep2 = _aggregate(events, ['event_player_2_id', 'event_team_abbr_2'] + list(second_group), second).rename({'event_player_2_id': 'player_id', 'event_team_abbr_2': 'team_abbr'}).with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    ep3 = _aggregate(events, ['event_player_3_id', 'event_team_abbr'] + list(second_group), {'secondary_assists': ('is_goal', 'sum')}).rename({'event_player_3_id': 'player_id', 'event_team_abbr': 'team_abbr'}).with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    indv = ep1.join(ep2, on=clean, how='full', coalesce=True).join(ep3, on=clean, how='full', coalesce=True)
    shot_mapping = {k: v for k, v in first.items() if k in ('goals', 'shots', 'fenwick', 'corsi', 'expected_goals')}
    # Aggregate all shot types in one group-by.  The previous implementation
    # scanned and joined the event frame once per shot type.
    shot_keys = ['event_player_1_id', 'event_team_abbr'] + list(second_group)
    shot_exprs = []
    for shot_type in SHOT_TYPES:
        prefix = shot_type.replace('-', '_')
        for name, (source, _) in shot_mapping.items():
            shot_exprs.append(
                pl.when(pl.col('shot_type') == shot_type)
                .then(pl.col(source))
                .otherwise(0)
                .sum()
                .alias(f'{prefix}_{name}')
            )
    shots = events.filter(pl.col('shot_type').is_in(SHOT_TYPES)).group_by(shot_keys, maintain_order=True).agg(shot_exprs)
    shots = shots.rename({'event_player_1_id': 'player_id', 'event_team_abbr': 'team_abbr'}).with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    indv = indv.join(shots, on=clean, how='full', coalesce=True)
    indv = indv.with_columns([pl.col(c).fill_null(0) for c in ['goals', 'primary_assists', 'secondary_assists', 'penalties_taken', 'penalties_drawn', 'faceoff_wins', 'faceoff_losses'] if c in indv.columns])
    return indv.with_columns([(pl.col('goals') + pl.col('primary_assists')).alias('primary_points'), (pl.col('goals') + pl.col('primary_assists') + pl.col('secondary_assists')).alias('points')])


def _combined(pbp, group, game_strength, second_group):
    frames = [process_stats(pbp, group, venue, game_strength, second_group) for venue in ('away', 'home')]
    if group == 'skater':
        keys = ['player_id', 'team_abbr', 'season'] + (['game_id'] if 'game_id' in second_group else [])
    elif group == 'goalie':
        keys = ['player_id', 'team_abbr'] + list(second_group)
    else:
        keys = ['team_abbr'] + list(second_group)
    cols = [c for c in frames[0].columns if c not in keys]
    out = _aggregate(pl.concat(frames, how='vertical_relaxed'), keys, {c: (c, 'sum') for c in cols})
    if group == 'skater':
        out = out.with_columns([(pl.col('primary_fenwick_assists') + pl.col('secondary_fenwick_assists')).alias('fenwick_assists'), (pl.col('primary_expected_assists') + pl.col('secondary_expected_assists')).alias('expected_assists')])
    return out.with_columns((pl.col('expected_goals_against') - pl.col('goals_against')).alias('goals_saved_above_expected'))


def calc_onice(pbp, game_strength, second_group):
    return _combined(pbp, 'skater', game_strength, second_group)


def calc_team(pbp, game_strength, second_group):
    return _combined(pbp, 'team', game_strength, second_group)


def calc_goalie(pbp, game_strength, second_group):
    return _combined(pbp, 'goalie', game_strength, second_group)


def rank_stats(df, rates=True, comparison=True, group_by=None):
    base_columns = list(df.columns)
    df = df.with_columns(
        (
            pl.when(pl.col('position').is_in(['C', 'L', 'R']))
            .then(pl.lit('F'))
            .otherwise(pl.col('position'))
            .fill_null(0)
            if 'position' in df.columns else pl.lit('team')
        ).alias('head_position')
    )
    groups = [c for c in (group_by or []) if c not in ['player_id', 'player_name', 'position', 'team_abbr']] + ['head_position']
    if not rates and not comparison:
        return df.drop('head_position')
    rate_exprs = []
    percentile_specs = []
    generated_order = []
    for stat in list(df.columns):
        if stat in BIO_STAT_COL + ['player_id', 'season', 'time_on_ice', 'position', 'position_group'] or not df[stat].dtype.is_numeric():
            continue
        penalty = 'penalties' in stat or 'penalty_minutes' in stat
        good = 'drawn' in stat or stat.endswith('_against')
        invert = ('against' in stat and not (penalty and stat.endswith('_against'))) or (penalty and not good) or stat == 'giveaways'
        if not any(s in stat for s in NON_TOTALS):
            per = f'{stat}_per_sixty'
            if rates:
                rate_exprs.append((pl.col(stat) / pl.col('time_on_ice') * 60).alias(per))
                generated_order.append(per)
                if comparison:
                    percentile_specs.append((per, invert))
                    generated_order.append(f'{per}_percentile')
        if comparison and any(s in stat for s in SPECIAL_KEYS):
            percentile_specs.append((stat, invert))
            generated_order.append(f'{stat}_percentile')
    if rate_exprs:
        df = df.with_columns(rate_exprs)
    if comparison:
        df = df.with_columns([
            ((1 - pl.col(stat).rank('average').over(groups) / pl.col(stat).count().over(groups)) if invert else (pl.col(stat).rank('average').over(groups) / pl.col(stat).count().over(groups))).alias(f'{stat}_percentile')
            for stat, invert in percentile_specs
            if stat in df.columns or any(expr.meta.output_name() == stat for expr in rate_exprs)
        ])
    if generated_order:
        df = df.select(base_columns + ['head_position'] + [column for column in generated_order if column in df.columns])
    return df.drop('head_position')


def _metric_expr(expr, columns):
    return eval(expr, {'__builtins__': {}}, {c: pl.col(c) for c in columns})


def extra_calc(df, metrics):
    for new_col, num_expr, denom_expr in AGG_POST_METRICS + metrics:
        try:
            if not (num_expr in df.columns or any(op in num_expr for op in '+-*/')):
                continue
            num = _metric_expr(num_expr, df.columns)
            if denom_expr and (denom_expr in df.columns or any(op in denom_expr for op in '+-*/')):
                expr = (num / _metric_expr(denom_expr, df.columns).replace(0, None)).fill_null(0)
            else:
                expr = num.fill_null(0)
            df = df.with_columns(expr.alias(new_col))
        except (KeyError, NameError, SyntaxError):
            pass
    return df


def apply_rosters(df, group, schedule_path, roster_path):
    if group == 'team':
        schedule = _read_csv_cached(schedule_path).filter(~pl.col('game_state').is_in(NON_FINAL_STATES)).with_columns([(pl.col('home_score') > pl.col('away_score')).cast(pl.Int64).alias('home_win'), (pl.col('away_score') > pl.col('home_score')).cast(pl.Int64).alias('away_win'), (pl.col('period_type_last') == 'REG').cast(pl.Int64).alias('regulation'), (pl.col('period_type_last') == 'OT').cast(pl.Int64).alias('overtime'), (pl.col('period_type_last') == 'SO').cast(pl.Int64).alias('shootout')])
        frames = []
        for venue in ('home', 'away'):
            standing = schedule.select(['game_id', f'{venue}_team_abbr', f'{venue}_win', 'regulation', 'overtime', 'shootout'])
            states = ('regulation', 'overtime', 'shootout')
            standing = standing.with_columns([
                expression
                for state in states
                for expression in (
                    pl.when(pl.col(state) == 1).then(pl.col(f'{venue}_win')).otherwise(0).alias(f'{state}_wins'),
                    pl.when(pl.col(state) == 1).then(1 - pl.col(f'{venue}_win')).otherwise(0).alias(f'{state}_losses'),
                )
            ])
            frames.append(standing.drop([f'{venue}_win', 'regulation', 'overtime', 'shootout']).rename({f'{venue}_team_abbr': 'team_abbr'}))
        return df.join(pl.concat(frames, how='vertical_relaxed'), on=['game_id', 'team_abbr'], how='left').with_columns((pl.col('team_abbr') + pl.col('season').cast(pl.String)).alias('wsba_id'))
    rosters = _read_csv_cached(roster_path)
    names = rosters.select(['player_id', 'player_name', 'headshot', 'position', 'handedness', 'height_in', 'weight_lbs', 'birth_date', 'birth_country']).unique('player_id', keep='last')
    df = df.with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    names = names.with_columns(pl.col('player_id').cast(pl.Int64, strict=False))
    remove = [c for c in df.columns if c in names.columns and c != 'player_id']
    complete = df.drop(remove, strict=False).join(names, on='player_id', how='left').with_columns([pl.col('birth_date').cast(pl.String).str.to_date(strict=False).alias('birth_date'), pl.col('season').cast(pl.String).str.slice(4, 4).cast(pl.Int64, strict=False).alias('season_year')]).with_columns([(pl.col('season_year') - pl.col('birth_date').dt.year()).alias('age'), ('https://assets.nhle.com/mugs/nhl/' + pl.col('season').cast(pl.String) + '/' + pl.col('team_abbr').cast(pl.String) + '/' + pl.col('player_id').cast(pl.Int64, strict=False).cast(pl.String) + '.png').alias('headshot'), (pl.col('player_id').cast(pl.String) + pl.col('season').cast(pl.String) + pl.col('team_abbr').cast(pl.String)).alias('wsba_id')])
    if group == 'skater':
        # Keep rows with missing roster positions because null is not ``'G'``.
        complete = complete.filter(pl.col('position').is_null() | (pl.col('position') != 'G'))
    elif group == 'game_score':
        complete = complete.filter(pl.col('position').is_null() | (pl.col('position') != 'G') | pl.col('points').is_null())
    return complete


def apply_params(df, group_by, params, stage='before'):
    for col, spec in params.items():
        op = spec[0]
        timing = spec[-1] if spec[-1] in ('before', 'after') else 'before'
        vals = list(spec[1:-1] if spec[-1] in ('before', 'after') else spec[1:])
        if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
            vals = list(vals[0])
        if timing != stage:
            continue
        if op == 'last':
            keys = group_by if isinstance(group_by, list) else [group_by]
            df = df.sort(keys + [col]).with_columns(pl.int_range(0, pl.len()).alias('__row'))
            df = df.with_columns(pl.col('__row').cum_count().over(keys).reverse().alias('__keep')).filter(pl.col('__keep') <= vals[0]).drop(['__row', '__keep'])
        else:
            target = pl.col(col)
            if 'date' in col.lower():
                target = target.cast(pl.String).str.to_datetime(strict=False)
                vals = [pl.Series([v]).str.to_datetime(strict=False).item() for v in vals]
            if op == 'between':
                mask = target.is_between(vals[0], vals[1])
            elif op == 'in':
                mask = target.is_in(vals)
            elif op == 'not in':
                mask = ~target.is_in(vals)
            else:
                mask = OPS[op](target, *vals)
            df = df.filter(mask.fill_null(False))
    return df
