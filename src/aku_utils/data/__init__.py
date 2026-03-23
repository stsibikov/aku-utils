import logging
import warnings
from typing import Any, Literal

import pandas as pd

from aku_utils.utils import process_cols_input

logger = logging.getLogger(__name__)


def drop(
    df: pd.DataFrame, cols: list[str] | str, if_no_column: Literal["raise", "warn", "skip"] = "warn"
):
    """
    Args
    ---
        cols:
            cols to drop.
            if string with no tabs, treats it as one column.
            if string with tabs, separates the string by tabs
            and uses it as list of strings.
    """
    cols = process_cols_input(cols)

    missing_cols = list(set(cols) - set(df.columns))
    if missing_cols:
        message = f"no columns `{missing_cols!s}` in df."
        if if_no_column == "raise":
            raise ValueError(message)
        elif if_no_column == "warn":
            warnings.warn(message, stacklevel=2)

    cols_set = set(cols) - set(missing_cols)
    df = df.drop(columns=list(cols_set))
    return df


def coalesce(df: pd.DataFrame, cols: list[str] | str, value: Any | None = None) -> pd.Series:
    """
    Args
    ---
        cols:
            cols to coalesce.
            if string with no tabs, treats it as one column.
            if string with tabs, separates the string by tabs
            and uses it as list of strings.
        value:
            nulls (after column coalescing) will be replaced with `value`.
    """
    cols = process_cols_input(cols)

    srs = df[cols].bfill(axis=1).iloc[:, 0]

    if value is not None:
        srs = srs.fillna(value)

    return srs


def reorder(
    df: pd.DataFrame,
    value: dict[str, int] | list[str],
    if_no_column: Literal["raise", "warn", "skip"] = "raise",
) -> pd.DataFrame:
    """
    Reorder dataframe columns.

    Args
    ---
    value: dict[str, int], list[str]
        if `dict[str, int]`, a column (key) will be put in specified
        space (value, 0-indexed).

        if `list[str]`, puts the columns in the list in the front
    """

    def reorder_dict(df: pd.DataFrame, value: dict[str, int], if_no_column):
        """
        we put the target cols in their spots. then we fill the rest
        with cols that are not target.
        """
        missing_cols = []
        for key in list(value):
            if key not in df.columns:
                missing_cols.append(key)
                value.pop(key)

        if missing_cols and if_no_column in ("raise", "warn"):
            msg = f"Columns `{missing_cols}` not found in df"
            if if_no_column == "raise":
                raise ValueError(msg)
            else:
                warnings.warn(msg, stacklevel=2)

        pos_set = set()
        for pos in value.values():
            if pos in pos_set:
                msg = f"value contains duplicate positions: `{pos}`"
                raise ValueError(msg)
            else:
                pos_set.add(pos)

        old_cols = list(df.columns)
        left_cols = [col for col in old_cols if col not in value]
        new_cols = [None] * len(old_cols)
        for col, pos in value.items():
            new_cols[pos] = col

        for i, col in enumerate(new_cols):
            if col is None:
                new_cols[i] = left_cols.pop(0)

        df = df[new_cols]
        return df

    def reorder_list(df: pd.DataFrame, value: list[str], if_no_column):
        missing_cols = [col for col in value if col not in df.columns]

        if missing_cols and if_no_column in ("raise", "warn"):
            msg = f"Columns `{missing_cols}` not found in df"
            if if_no_column == "raise":
                raise ValueError(msg)
            else:
                warnings.warn(msg, stacklevel=2)

        value = [col for col in value if col in df.columns]
        left_cols = [col for col in df.columns if col not in value]
        new_cols = value + left_cols

        df = df[new_cols]
        return df

    if isinstance(value, dict):
        df = reorder_dict(df, value, if_no_column)
    elif isinstance(value, list):
        df = reorder_list(df, value, if_no_column)
    else:
        msg = f"value must be dict or list, not `{type(value)}`"
        raise TypeError(msg)
    return df


def dedupe(
    df: pd.DataFrame,
    subset: list[str] | int | None,
    return_dict: bool = False,
    **kwargs,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    If duplicates are found, and `return_dict` is True, then they will
    be stored in the result dict, otherwise they will be logged.

    Args
    ---
    subset:
        if int, first `subset` number of columns are used as actual subset
    kwargs:
        for `df.duplicated()`

    Returns
    ---
    dict `{'df': dataframe, 'dupes': duplicates}` if `return_dict` is True.

    dataframe if `return_dict` is False

    Usage
    ---
    ```
    res: dict = dedupe(df, 2, return_dict=True)
    df: pd.DataFrame = res['df']
    dupes: pd.DataFrame = res['dupes']


    df: pd.DataFrame = dedupe(df, 2)
    ```
    """
    if isinstance(subset, int):
        subset = df.columns[:subset]

    duped_srs = df.duplicated(subset, **kwargs)
    order = df.columns if subset is None else subset
    duped_df = df[duped_srs].sort_values(order)
    deduped_df = df[~duped_srs]

    if duped_srs.any():
        logger.info(f"duplicates found:\n{duped_df.head(5)}")

    if return_dict:
        res = {}
        res["dupes"] = duped_df
        res["df"] = deduped_df
    else:
        res = deduped_df

    return res


def dmerge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["inner", "left", "outer"] = "outer",
    on: list[str] | int | None = None,
    suffixes: tuple[str, str] = ("", "_1"),
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Detailed merge. Will return merged dataframe
    and parts that were not matched.

    Args
    ---
        on:
            if int, first `on` left columns are merging columns

    Considerations
    ---
        does not work with `left_on` `right_on`.

    Returns
    ---
    dict with key-value pairs
        df: merged dataframe

        left_only: left dataframe with unmatched rows

        right_only: right dataframe with unmatched rows
    """
    res: dict[str, pd.DataFrame] = {}
    left_flag = "__dmerge_left"
    right_flag = "__dmerge_right"

    left = left.copy()
    right = right.copy()

    left_original_cols = list(left.columns)
    right_original_cols = list(right.columns)

    if isinstance(on, int):
        on = list(left.columns)[:on]
    elif on is None:
        on = list(set(left.columns) & set(right.columns))

    # merge preparation
    left[left_flag] = 1.0
    right[right_flag] = 1.0

    df = left.merge(right, how="outer", on=on, suffixes=suffixes, **kwargs)

    # defining left_only segment
    # needs to be processed to obtain
    # original look
    left_only = df[(df[left_flag] == 1) & (df[right_flag] != 1)].drop(
        columns=[left_flag, right_flag]
    )

    # processing the segment

    def get_left_schema(left_cols, right_cols, on, left_suffix) -> dict[str, str] | None:
        """
        looks at columns who had their names changed by merge.
        makes a dict with new (after merge) name to old (original) name.
        """
        if left_suffix == "":
            return None

        new_old_map = {}
        for col in left_cols:
            if col in right_cols and col not in on:
                new_old_map[col + left_suffix] = col

        return new_old_map

    new_old_map = get_left_schema(
        left_cols=left_original_cols, right_cols=right_original_cols, on=on, left_suffix=suffixes[0]
    )

    if new_old_map is not None:
        left_only = left_only.drop(
            # dropping columns that were not
            # left originally, but had their names
            # changed to ones that match original
            # left columns
            columns=new_old_map.values()
        ).rename(
            # once "impostor" columns are gone
            # we can change new (after merge) columns
            # that got the suffix to old (original) names
            columns=new_old_map
        )[
            # after restoring the names,
            # we use original positioning to
            # restore the columns order
            # and remove right columns we got after merge
            left_original_cols
        ]
    else:
        # names were not changed, so the only thing
        # needed is to restore original order
        left_only = left_only[left_original_cols]

    # done processing
    res["left_only"] = left_only

    if left_only.shape[0] > 0:
        logger.info(f"left only rows found:\n{left_only.head(5)}")

    # defining right_only segment
    # and performing similar operations
    right_only = df[(df[left_flag] != 1) & (df[right_flag] == 1)].drop(
        columns=[left_flag, right_flag]
    )

    # processing the segment

    def get_right_schema(left_cols, right_cols, on, right_suffix) -> dict[str, str] | None:
        """
        looks at columns who had their names changed by merge.
        makes a dict with new (after merge) name to old (original) name.
        """
        if right_suffix == "":
            return None

        new_old_map = {}
        for col in right_cols:
            if col in left_cols and col not in on:
                new_old_map[col + right_suffix] = col

        return new_old_map

    new_old_map = get_right_schema(
        left_cols=left_original_cols,
        right_cols=right_original_cols,
        on=on,
        right_suffix=suffixes[1],
    )

    # refer to similar processing in left_only
    if new_old_map is not None:
        right_only = right_only.drop(columns=new_old_map.values()).rename(columns=new_old_map)[
            right_original_cols
        ]
    else:
        right_only = right_only[right_original_cols]

    # done processing
    res["right_only"] = right_only

    if right_only.shape[0] > 0:
        logger.info(f"right only rows found:\n{right_only.head(5)}")

    # postprocessing final df
    if how == "inner":
        df = df[(df[left_flag] == 1) & (df[right_flag] == 1)]
    elif how == "left":
        df = df[(df[left_flag] == 1)]
    else:
        pass

    res["df"] = df.drop(columns=[left_flag, right_flag])
    return res
