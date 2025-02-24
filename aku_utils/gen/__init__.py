import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from aku_utils.common import curr_date

def gen_effects(
    n_periods : int,
    num_objs : int,
    window : int = 5,
    double_smooth : bool = False,
):
    '''
    Generate dataframe with smoothed random standard noise

    Smoothness increases with `window`. It must not be equal or higher than `n_periods`
    - it will produce flat effects.
    '''
    if window >= n_periods:
        raise ValueError(f'window ({window}) must be lower than n_periods ({n_periods})')

    effects = np.random.normal(loc=0, scale=1, size=(n_periods, num_objs + 1))
    effects = pd.DataFrame(effects, columns=['base'] + list(range(num_objs)))
    effects = pd.concat([effects, effects.iloc[:(1 + double_smooth) * window]])

    effects = effects.rolling(window).mean()

    if double_smooth:
        effects = effects.rolling(window).mean()

    effects = effects.iloc[(1 + double_smooth) * window:, :].reset_index(drop=True)
    return effects

def panel_data(
    num_objs : int = 10,
    days_back : int = 90,
    full_df : bool = False,
    config : Optional[Dict] = None,
) -> pd.DataFrame:
    '''
    Generate hourly panel data with daily and weekly effects.

    Arguments
    ---
    num_objs:
        number of objects in the dataframe
    days_back:
        number of days past data is available for
    full_df:
        returns additional columns used in dataframe generation
    config:
        config for random noise affinity generation

    Config
    ---
    Want trends to not cross between different objects? Decrease trend scale
    Want more uniqueness? Decrease global effects scale

    Suggestion: you could redo `np.random.gamma` function to supply
    it with `mean` and `var`, then set `a` and `b` to `mean**2/var`
    and `var/mu` accordingly

    Usage
    ---
    ```python
    from aku_utils.gen import panel_data
    df = panel_data()

    # Inspect
    import plotly.express as px
    dfv = df.pivot(
        index='dt',
        columns='obj_id',
        values='target'
    ).reset_index()

    px.line(
        dfv,
        x='dt',
        y=[c for c in dfv.columns if c not in ['dt']]
    )
    ```
    '''

    cap_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    objs_ids = list(range(num_objs))
    obj_names = [''.join(np.random.choice(list(cap_letters), size=3)) for i in range(num_objs)]

    if config is None:
        config = {
            'trend' : {
                'global' : {'alpha' : 2**5, 'beta' : 0.5},
                'local' : {'alpha' : 2**9, 'beta' : 1},
            },
            'weekly' : {
                'global' : {'alpha' : 1, 'beta' : 2},
                'local' : {'alpha' : 4, 'beta' : 4},
            },
            'daily' : {
                'global' : {'alpha' : 0.5, 'beta' : 4},
                'local' : {'alpha' : 1, 'beta' : 8},
            },
            'min_scale' : 0.1,
            'min_start' : -100,
            'max_start' : 100,
        }

    objs_base = {
        'obj_id' : objs_ids,
        'obj' : obj_names,
        'start' : np.random.uniform(
            low=config['min_start'],
            high=config['max_start'],
            size=num_objs),
        'trend_global_scale' : np.random.gamma(
            shape=config['trend']['global']['alpha'],
            scale=config['trend']['global']['beta'],
            size=num_objs) + config['min_scale'],
        'trend_local_scale' : np.random.gamma(
            shape=config['trend']['local']['alpha'],
            scale=config['trend']['local']['beta'],
            size=num_objs) + config['min_scale'],
        'weekly_global_scale' : np.random.gamma(
            shape=config['weekly']['global']['alpha'],
            scale=config['weekly']['global']['beta'],
            size=num_objs) + config['min_scale'],
        'weekly_local_scale' : np.random.gamma(
            shape=config['weekly']['local']['alpha'],
            scale=config['weekly']['local']['beta'],
            size=num_objs) + config['min_scale'],
        'daily_global_scale' : np.random.gamma(
            shape=config['daily']['global']['alpha'],
            scale=config['daily']['global']['beta'],
            size=num_objs) + config['min_scale'],
        'daily_local_scale' : np.random.gamma(
            shape=config['daily']['local']['alpha'],
            scale=config['daily']['local']['beta'],
            size=num_objs) + config['min_scale'],
    }
    objs = pd.DataFrame(objs_base)

    date_range = pd.date_range(
        start = curr_date - timedelta(days=days_back),
        end = curr_date + timedelta(days=3, hours=23),
        freq = 'h'
    )

    trend_effects = gen_effects(
        n_periods=date_range.shape[0],
        num_objs=num_objs,
        window=128,
        double_smooth=True
    )


    trend_effects = trend_effects.assign(**{
        'dt' : date_range
    })

    trend_effects = trend_effects.melt(
        id_vars='dt',
        value_vars=[c for c in trend_effects if c not in ['dt']],
        var_name='obj_id',
        value_name='trend',
    )
    df = objs.merge(trend_effects, on='obj_id')

    df = df.merge(
        trend_effects.loc[trend_effects['obj_id'] == 'base', ['dt', 'trend']],
        on='dt',
        suffixes=('', '_base')
    )
    df = df.assign(**{
        'hour' : df['dt'].dt.hour,
        'day_of_week' : df['dt'].dt.day_of_week
    })
    daily_effects = gen_effects(n_periods=24, num_objs=num_objs, window=5)


    daily_effects = daily_effects.reset_index(names='hour')

    daily_effects = daily_effects.melt(
        id_vars='hour',
        value_vars=daily_effects.columns,
        var_name='obj_id',
        value_name='daily_effect',
    )
    df = df.merge(
        daily_effects,
        on=['obj_id', 'hour']
    )
    df = df.merge(
        daily_effects.loc[daily_effects['obj_id'] == 'base', ['hour', 'daily_effect']],
        on='hour',
        suffixes=('', '_base')
    )
    weekly_effects = gen_effects(n_periods=7, num_objs=num_objs, window=3)

    weekly_effects = weekly_effects.reset_index(names='day_of_week')

    weekly_effects = weekly_effects.melt(
        id_vars='day_of_week',
        value_vars=weekly_effects.columns,
        var_name='obj_id',
        value_name='weekly_effect',
    )
    df = df.merge(
        weekly_effects,
        on=['obj_id', 'day_of_week']
    )

    df = df.merge(
        weekly_effects.loc[weekly_effects['obj_id'] == 'base', ['day_of_week', 'weekly_effect']],
        on='day_of_week',
        suffixes=('', '_base')
    )
    df['target'] = (
        df['start'] +
        df['trend_local_scale'] * df['trend'] + df['trend_global_scale'] * df['trend_base'] +
        df['daily_local_scale'] * df['daily_effect'] + df['daily_global_scale'] * df['daily_effect_base'] +
        df['weekly_local_scale'] * df['weekly_effect'] + df['weekly_global_scale'] * df['weekly_effect_base']
    )

    if not full_df:
        df = df[['obj_id', 'obj', 'dt', 'target']]

    return df

