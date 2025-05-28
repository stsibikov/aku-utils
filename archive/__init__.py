from aku_utils import unnest_dict_els
from typing import Sequence
import pandas as pd
import numpy as np


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


def wmavg(
    arr,
    weights,
    wrap: bool = False,
):
    '''
    Calculate weighted moving average. For element at position `i`, its
    weighted average will be:
    `arr[i] * weights[0] + arr[i-1] * weights[1] + arr[i-2] * weights[2] + ...`

    Parameters
    ---
    wrap:
        whether to wrap the array around.
        For example, if array is [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
        and weights are [.7, .2, .1], 1st element in the output is
        `1 * .7 + 10 * .2 + 9 * .1`

    Returns
    ---
    array of same length as the starting one
    '''
    if wrap:
        # number of elements that will be moved back and forth to
        # simulate array being wrapped around
        n_pop_elements = len(weights) - 1
        # add starting elements to end of the array
        arr = np.concat([arr, arr[:n_pop_elements]])

    avg = np.convolve(arr, weights, mode='full')[:len(arr)]

    if wrap:
        # concat 1) elements that were copied to the end
        # of the array and 2) original array without the starting
        # elements
        avg = np.concat([
            avg[-n_pop_elements:],  # type: ignore
            avg[n_pop_elements:-n_pop_elements]  # type: ignore
        ])

    return avg
