'''
A utils module that contains light weight utils for type wrangling.
Used by other modules
'''

from datetime import datetime
from typing import Dict, Sequence, Hashable

epsilon = 1e-5

today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

# the format is int({year}{week_of_year})
this_week = int(today.strftime('%Y%V'))


def is_iter(v):
    '''check if value is an iterable, with string not being counted as an iterable'''
    return hasattr(v, '__iter__') and not isinstance(v, str)


def to_list(v):
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


def manipulate_args(
    names : Sequence[str]
):
    '''
    Example decorator that manipulates function arguments
    '''

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            from inspect import signature

            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            # dict of all arguments passed to the function
            f_args = bound_args.arguments

            # the bind produces {'a' : 1, 'kwargs' : {'b' : 2}}
            # instead of intended {'a' : 1, 'b' : 2}
            unnest_dict_els(f_args, 'kwargs')

            # parameter processing
            # for parameter in f_args:
            #     if parameter in names and isinstance(parameter, int):
            #         f_args[parameter] = f_args[parameter] + 1

            return func(**f_args)
        return wrapper
    return decorator
