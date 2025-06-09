'''
Commonly used variables and functions
'''

from datetime import date, datetime
from typing import Dict, List, Sequence, Hashable

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
