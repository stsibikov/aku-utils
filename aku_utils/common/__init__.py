'''
Commonly used variables and functions
'''
import pandas as pd
from datetime import date
from typing import Dict, List, Sequence, Hashable, Any

epsilon = 1e-5

today = date.today()

# iso compliant {year}{week} int
this_week = int(today.strftime('%G%V'))


def is_iter(obj) -> bool:
    '''
    check if value is an iterable, with string not being counted as an iterable
    '''
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def is_sequence_of_str(obj):
    return (
        is_iter(obj) and all(isinstance(item, str) for item in obj)
    )


def to_list(v) -> List:
    if not is_iter(v):
        return [v]
    return list(v)


def unnest_dict_els(dct : Dict, els : Sequence[Hashable]) -> None:
    '''
    In-place operation to unnest dictionary elements.
    '''
    els = to_list(els)
    for el in els:
        if el in dct:
            unnested_content = dct.pop(el)
            dct.update(unnested_content)
    return


def pdisplay(obj: Any) -> None:
    '''protected display'''
    try:
        display(obj)  # type: ignore
    except NameError:
        print(obj)
    return None


def newcol(df: pd.DataFrame, name) -> str:
    '''
    Returns a name for a new column.
    The function checks if the columns is
    already in the dataframe. If it is,
    it looks for column named "{name}_2".
    If it is there, it looks for a column
    with pattern "{name}_{number}" with the maximum
    number, and then returns "{name}_{number+1}"
    to obtain a unique name for a new column.
    '''

    cols = set(df.columns.to_list())

    if name not in cols:
        return name

    new_name = f'{name}_2'
    if new_name not in cols:
        return new_name

    import re
    pattern = re.compile(rf'{name}_(\d+)$')

    max_num = float('-inf')
    for col in cols:
        match = pattern.match(col)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    return f'{name}_{max_num+1}'
