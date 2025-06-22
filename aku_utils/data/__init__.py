'''
Data and analytical operations

Healthcheck operations convention:
* function returns and operates `msg` - message with 1) a final dataset 2)
other parameters and datasets that are essential to this healthcheck
* if one of the "essential datasets" are missing, for example, all rows
got their matches during a merge, then the function returns None instead
of empty dataframe for that part of the message. As such, these functions
return `dict[str, Union[pd.DataFrame, None]]`
* None instead of empty dataframe was chosen because operations on empty
dataframe can sometimes still run, while they should fail
'''
from typing import List, Literal, Union
from warnings import warn
import numpy as np
import pandas as pd

from aku_utils.common import pdisplay

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
    De-duplicate a dataframe

    A dataframe healthcheck function

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
        warn(f'{duplicated_share = }. Duplicates:')
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
            warn(f'{nulls_share = }. Rows with nulls:')
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


def cmerge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how = Literal['inner', 'left', 'outer'],
    on = List[str],
    display: bool = True,
) -> dict[str, Union[pd.DataFrame, None]]:
    '''
    Careful merge

    Merges and reports what rows from left and right
    dataframes did not get their match

    A dataframe healthcheck function

    Behavior
    ---
    This section aims to tackle possible confusion
    as to what `left_only` and `right_only` keys
    in the message will mean for different `how`
    argument

        `inner`:
            `left_only` and `right_only` contain
            rows that did not make their way into
            the final dataset
        `left`:
            `left_only` contains rows that did not
            get a match from the right dataset.
            `right_only` contains rows that did not
            make their way into the final dataset
        `outer`:
            `left_only` contains rows that did not
            get a match from the right dataset.
            `right_only` contains rows that did not
            get a match from the left dataset.

    '''
    # placeholder for return value
    msg: dict[str, Union[dict, pd.DataFrame, None]] = {}

    # copy is to prevent flags from appearing in input dfs
    left = left.copy()
    right = right.copy()

    # this is to understand what columns to drop
    # from `left_only` and `right_only` parts of the message
    # later on
    cols_set_left = set(left.columns.to_list())
    cols_set_right = set(right.columns.to_list())

    cols_exclusive_to_left = cols_set_left - cols_set_right
    cols_exclusive_to_right = cols_set_right - cols_set_left

    # flags are used to determine what rows
    # make it to final dataframe
    # and what rows go to `left_only` and `right_only`
    lf = 'cmerge_row_flag_left'
    rf = 'cmerge_row_flag_right'
    left[lf] = 1
    right[rf] = 1

    # every merge type is essentially outer merge
    # with optional filtering later
    df = pd.merge(left=left, right=right, how='outer', on=on)
    df[[lf, rf]] = df[[lf, rf]].fillna(0)

    match how:
        case 'inner':
            # df after filtering
            # we will pass it as final dataframe
            # original df is still needed to get left and
            # right only rows
            dfaf = df[(df[lf] == 1) & (df[rf] == 1)]
        case 'left':
            dfaf = df[(df[lf] == 1)]
        case 'outer':
            dfaf = df
        case _:
            raise ValueError(f'invalid `how` argument: {how}. See docs')

    # fill in `left_only` and `right_only` parts of the message
    for side in ['left', 'right']:
        is_left = side == 'left'
        other_side = 'left' if side == 'right' else 'right'

        # this is `left_only` or `right_only`
        this_side_only = df[
            (df[lf] == int(is_left)) & (df[rf] == int(not is_left))
        ].drop(columns=[lf, rf])

        cols_to_drop = (
            cols_exclusive_to_right if is_left else cols_exclusive_to_left
        )
        this_side_only = this_side_only.drop(columns=cols_to_drop)

        # notice the behavior
        # if dataframe is empty (no rows lost)
        # this part of message will be None
        # this is done because checking if `object is None`
        # is easier than checking if `object is empty dataframe`
        this_side_only_rows = this_side_only.shape[0]
        if this_side_only_rows > 0:
            warn(
                f'{side} dataframe: {this_side_only_rows} rows did not'
                f'get a match from the {other_side} one:'
            )

            if display:
                pdisplay(this_side_only)

            msg[f'{side}_only'] = this_side_only
        else:
            msg[f'{side}_only'] = None

    dfaf = dfaf.drop(columns=[lf, rf])
    msg['df'] = dfaf
    return msg
