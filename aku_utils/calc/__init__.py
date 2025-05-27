from aku_utils import to_list, this_week, today, epsilon
import pandas as pd
import numpy as np
from typing import Optional, Sequence


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

    avg : pd.Series = (df['weighted_value'] / df[weight]).rename(name)

    if return_df:
        avg = avg.to_frame().reset_index()  # type: ignore

    return avg


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

    avg = wavg(
        df,
        value,
        'weight_for_wmean',
        group,
        return_df=return_df,
        name=name
    )
    return avg


class BiasFunc:
    def __init__(self, point1, point2, a) -> None:
        '''
        Implementation of Barron 2020 bias function, may be slightly modified

        [Link to plot](https://www.desmos.com/calculator/cybgxv3yd0) of the function.

        Example
        ---
        ### Smoothing
        Suppose we want to create a flexible array smoothing function based on
        weighted moving average.

        We want to be able to input just one number, which represents how smooth do we
        want the array to be, and the bias function outputs the width of the window for
        the weighted moving average, and the weights for each of the elements in the window.

        We can do this using this class. We set the first point to (1, 1) and second to
        (6, 0) - the X of the second point corresponds to the maximum possible length
        of the window. We will be interpolating between this points with the bias function,
        and extracting 1st, 2nd, 3rd and so on weights by getting the value of the bias function
        at 1, 2, 3 and so on.

        The bias function will interpolate between (1, 1) and (6, 0), so this
        means that, at maximum, we will be using 5 elements in our window.

        Finally, we set alpha (`a`) to a number between 0 and 1 (both non-inclusive).
        Numbers closer to 0 means that the array will be smoother, numbers closer
        to 1 means the array will retain more of the original shape, and alpha
        equal to .5 means that the interpolation between (1, 1) and (6, 0) will be linear.

        This is implemented in WindowWeights (same module)
        '''
        self.point1 = point1
        self.point2 = point2
        self.a = a

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        forbidden = [0, 1]
        if value in forbidden:
            raise ValueError(f"Attribute `a` cannot be part of: {forbidden}")
        self._a = value

    def __call__(self, value: float | Sequence[float]):
        x0 = self.point1[0]
        y0 = self.point1[1]

        x1 = self.point2[0]
        y1 = self.point2[1]

        a = self.a

        # just see the link in the doc string to see the function
        res = ((y1 - y0) / (x1 - x0) * (value - x0)) / (
            (1 / a - 2) * (1 - (value - x0) / (x1 - x0)) + 1
        ) + y0
        return res


def get_window_weights(
    a: float = 0.5,
    n: int = 5,
    normalize: bool = True
) -> np.ndarray:
    bias_func = BiasFunc(point1=(0, 1), point2=(n, 0), a=a)
    w = bias_func.__call__(np.arange(n))  # type: ignore
    w = w[w > 0]

    if normalize:
        w = w / w.sum()

    return w


def wmavg(
    arr,
    weights,
    wrap_back: bool = False,
):
    '''
    Calculate weighted moving average. For element at position `i`, its
    weighted average will be:
    `arr[i] * weights[0] + arr[i-1] * weights[1] + arr[i-2] * weights[2] + ...`

    Parameters
    ---
    wrap_back:
        whether to wrap the array around.
        For example, if array is [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
        and weights are [.7, .2, .1], 1st element in the output is
        `1 * .7 + 10 * .2 + 9 * .1`

    Returns
    ---
    array of same length as the starting one
    '''
    if wrap_back:
        # number of elements that will be moved back and forth to
        # simulate array being wrapped around
        n_pop_elements = len(weights)-1
        # add starting elements to end of the array
        arr = np.concat([arr, arr[:n_pop_elements]])

    avg = np.convolve(arr, weights, mode='full')[:len(arr)]

    if wrap_back:
        # concat 1) elements that were copied to the end
        # of the array and 2) original array without the starting
        # elements
        avg = np.concat([
            avg[-n_pop_elements:],  # type: ignore
            avg[n_pop_elements:-n_pop_elements]  # type: ignore
        ])

    return avg
