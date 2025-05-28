from aku_utils import to_list, epsilon
import pandas as pd
import numpy as np
from typing import Optional, Sequence

from .datetime import datetime_functions, datetime_base_features


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
    '''
    Creates window weights by using a bias function

    Weights are decreasing and positive. Increasing `n`
    makes the weights smoother, ie the change between this and the
    next weight is smaller.

    Decreasing `a` will produce weights that, when applied to an array for
    a weighted moving average, will produce a smoother array.
    '''
    bias_func = BiasFunc(point1=(0, 1), point2=(n, 0), a=a)
    w = bias_func.__call__(np.arange(n))  # type: ignore
    w = w[w > 0]

    if normalize:
        w = w / w.sum()

    return w


def wmavg(
    df : pd.DataFrame,
    value : str | Sequence[str],
    group : str | Sequence[str],
    weights : Sequence[float],
    wrap : bool = False
) -> pd.Series:
    '''
    Computes a weighted moving average on a pandas dataframe per group

    Arguments
    ---
        wrap:
            inserts end of groups into their starts before
            calculating the average to get averages where
            starts of groups look like the ends. Useful for
            generating random effects for cyclical series, like
            days of week, hours of day and so on.

    Considerations
    ---
    * Sort appropriately before using.
    * Weights are not internally normalized

    Algorithm
    ---
    Creates lagged columns based on length of weights
    and then calculates a weighted average in a row like usual
    '''
    from aku_utils.transforms.configs import flatten, finalize
    from aku_utils.transforms import Lags
    group = to_list(group)
    value = to_list(value)

    df = df[group + value]

    if wrap:
        # add end of series to the start

        # how many rows per group do we need to add
        n_add_rows = len(weights) - 1
        # add last n_add_rows to the start of each group
        # the order is preserved 
        df = pd.concat([
            df.groupby(group).tail(n_add_rows),
            df
        ])

    lag_config = flatten(
        {'c' : value, 't' : 'lag', 'l' : range(1, len(weights))}
    )

    lag_config = finalize(lag_config)

    lag_transformer = Lags(
        config=lag_config,
        group=group
    )

    # adds lagged columns
    df = lag_transformer.fit_transform(df)  # type: ignore

    if wrap:
        # delete those ends of groups we added earlier

        # how many rows in total we have to remove
        # all of them are at the start so removing them is no problem
        n_delete_rows = n_add_rows * df[group].drop_duplicates().shape[0]  # type: ignore
        df = df.iloc[n_delete_rows:, :]
    else:
        # fills nans in lagged (and value) columns with 0
        # to prevent rows from resulting to nan
        # also thats just how a window mean/sum/whatever works
        df.iloc[:, len(group):] = df.iloc[:, len(group):].fillna(0)

    # columns to perform multiplication on
    mult_col_list = value + [c['name'] for c in lag_config]

    # get a weighted average per row
    avg = (
        df[mult_col_list] * weights
    ).sum(axis=1)

    return avg
