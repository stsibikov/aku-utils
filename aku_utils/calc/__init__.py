from aku_utils import to_list, this_week, today, epsilon
import pandas as pd
import numpy as np
from typing import Optional


datetime_functions = {
    'year' : lambda srs: srs.dt.year,
    'month' : lambda srs: srs.dt.month,
    'day' : lambda srs: srs.dt.day,
    'hour' : lambda srs: srs.dt.hour,
    'minute' : lambda srs: srs.dt.minute,
    'second' : lambda srs: srs.dt.second,
    'day_of_week' : lambda srs: srs.dt.dayofweek,  # Monday=0, Sunday=6
    'quarter' : lambda srs: srs.dt.quarter,
    'is_month_start' : lambda srs: srs.dt.is_month_start,
    'is_month_end' : lambda srs: srs.dt.is_month_end,
    'is_quarter_start' : lambda srs: srs.dt.is_quarter_start,
    'is_quarter_end' : lambda srs: srs.dt.is_quarter_end,
    'is_year_start' : lambda srs: srs.dt.is_year_start,
    'is_year_end' : lambda srs: srs.dt.is_year_end,
    'day_of_year' : lambda srs: srs.dt.dayofyear,
    'week_of_year' : lambda srs: srs.dt.isocalendar().week.astype('int'),
    'timestamp' : lambda srs: srs.astype('int64') // 10**9,
    'date' : lambda srs: srs.dt.date,
    'time' : lambda srs: srs.dt.time,
    'is_leap_year' : lambda srs: srs.dt.is_leap_year,
    'week' : lambda srs: srs.dt.strftime('%Y%V').astype('int'),
    'relative_date' : lambda srs: (srs - today).dt.days,
    'relative_week' : lambda srs: (
        srs.dt.strftime('%Y%V').astype('int')
        - this_week
    )
}

# can be used by:
# df.assign(**{
#     name : ak.calc.datetime_functions[name](df['dt'])
#     for name in ak.calc.datetime_base_features
# })
datetime_base_features = [
    'year',
    'month',
    'day',
    'hour',
    'day_of_week',
    'day_of_year',
    'week_of_year',
    'timestamp',
    'week',
    'relative_date',
    'relative_week'
]


def points_to_coefs(x1, y1, x2, y2):
    '''
    Fit two points to y=ax+b equation. Returns a and b coefficients.

    Usage
    ---
    ```python
    point1 = (0, 1)
    point2 = (2, 3)
    a, b = points_to_coefs(*point1, *point2)
    ```
    '''
    if x1 == x2:
        raise ValueError('x1 and x2 cannot be equal')

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def sharify(
    df,
    values,
    group
) -> np.ndarray:
    '''
    Supports only positive values.

    If sum of a group is less or equal 0, then null is produced.
    Same if weight itself is null
    '''
    values_sum = df.groupby(group)[values].transform('sum')
    share = np.where(
        values_sum > 0,
        df[values] / values_sum,
        np.nan
    )
    return share


def wavg(
    df,
    value,
    weight,
    group,
    *,
    return_df : bool = False,
    name : Optional[str] = None
):
    df['weighted_value'] = df[value] * df[weight]
    df = df.groupby(group)[[weight, 'weighted_value']].sum()

    if not name:
        name = f'{value}_wavg'

    res : pd.Series = (df['weighted_value'] / df[weight]).rename(name)

    if return_df:
        res = res.to_frame().reset_index()  # type: ignore

    return res


def wmedian(
    df : pd.DataFrame,
    value,
    weight,
    group,
    *,
    return_df : bool = False,
    name : Optional[str] = None
):
    '''
    Weighted median
    '''
    group = to_list(group)
    if not name:
        name = f'{value}_wmedian'

    df = df.loc[
        df[weight] > 0,
        group + [value, weight]
    ]

    df['weight_norm'] = sharify(df, weight, group)
    df = df.drop(columns=weight)

    df = df.sort_values(group + [value])

    df['run_weight'] = df.groupby(group)['weight_norm'].cumsum()
    df = df.drop(columns='weight_norm')

    # to intercept situations where the weight doesnt land on exactly .5
    # we limit the result by 2 rows for each groups
    # the first is the one with the lowest `run_weight` above .5
    # the second is the one with the highest `run_weight` below .5
    # this way, we receive 2 rows for each group with weight closest to the desired one
    # if the closest weight is .5, we get two of the same rows
    all_low = df[
        # epsilon is required bc of floating point error ig
        # otherwise skips .5
        df['run_weight'] <= .5 + epsilon
    ].sort_values(
        group + ['run_weight'],
        ascending=[True] * len(group) + [False]
    )
    top_low = all_low.groupby(group).head(1)

    all_high = df[
        df['run_weight'] >= .5 - epsilon
    ].sort_values(
        group + ['run_weight']
    )
    top_high = all_high.groupby(group).head(1)

    df = pd.concat([top_low, top_high])

    # not needed tbh
    df = df.sort_values(group + [value])

    # after selecting 2 closest rows for each group, we
    # transform their weights for weighted mean calculation
    # the closer the `run_weight` to .5 the greater the weight must be
    df['weight_for_wmean'] = (df['run_weight'] - .5).abs() + epsilon
    df = df.drop(columns='run_weight')

    # non-zero cause we added epsilon.
    # the 'worst' weight in a group is 1 so
    # you can subtract .9 or something to intensify weights
    df['weight_for_wmean'] = (
        df.groupby(group)['weight_for_wmean'].transform('max')
        / df['weight_for_wmean']
    )

    res = wavg(
        df,
        value,
        'weight_for_wmean',
        group,
        return_df=return_df,
        name=name
    )
    return res
