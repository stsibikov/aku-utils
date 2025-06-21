'''
Data and analytical operations
'''
import numpy as np
import pandas as pd

from typing import List, Union
import warnings

from .datetime import datetime_features, datetime_base_features  # noqa
from .bias_and_weights import BiasFunc, get_window_weights  # noqa
from .averages import wavg, wmedian, wmavg  # noqa


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


def dedupe(
    df: pd.DataFrame,
    group: int | List[str] | None,
    *,
    allow_nulls: bool = False,
    display: bool = True,
    reset_index: bool = True,
) -> dict[str, Union[pd.DataFrame, None]]:
    '''
    De-duplicate a dataframe. Aimed to be a part of
    dataframe healthcheck inspection

    Beside just dropping the duplicates, this function
    extracts additional dataframes with duplicates and
    nulls (in identifying columns) and returns it.

    Parameters
    ---
        group:
            identifying columns.
            If `int`, then the first N columns
            are used as identifying columns.
        allow_nulls:
            whether to checks for nulls in identifying columns.
            Will remove rows with nulls in identifying columns
            from the final dataframe.
        reset_index:
            will reset index after clean up.
            It is only ever useful to set this to `False` if you
            are using your own index, and not the default integer
            range.
    '''
    from aku_utils import pdisplay
    msg: dict[str, Union[pd.DataFrame, None]] = {}

    if isinstance(group, int):
        group = df.columns[:group].to_list()

    # dataframe with all (ie both) the duplicates
    duplicated = df[df.duplicated(subset=group, keep=False)]

    duplicated_share = duplicated.shape[0] / df.shape[0]

    msg['duplicated'] = None

    if duplicated_share > 0:
        warnings.warn(f'{duplicated_share = }. Duplicates:')
        if group:
            duplicated = duplicated.sort_values(by=group)

        msg['duplicated'] = duplicated

        if display:
            pdisplay(duplicated)

    if not allow_nulls:
        msg['nulls'] = None

        nulls: pd.DataFrame = df[df[group].isna().any(axis=1)]  # type: ignore

        nulls_share = nulls.shape[0] / df.shape[0]

        if nulls_share > 0:
            warnings.warn(f'{nulls_share = }. Rows with nulls:')
            if group:
                nulls = nulls.sort_values(by=group)

            msg['nulls'] = nulls

            if display:
                pdisplay(nulls)

    # finally, we transform the original df
    df = df.drop_duplicates(subset=group, keep='first')

    if not allow_nulls:
        df = df[~df[group].isna().any(axis=1)]  # type: ignore

    if reset_index:
        df = df.reset_index(drop=True)

    msg['df'] = df

    return msg
