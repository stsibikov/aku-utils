'''
Module for generating and processing configs for transforms

# Main methods

flatten: multiply configs using a dict with lists as values
finalize: validate and process configs

'''


from typing import Union, List, Dict, Any
import warnings
from itertools import product
from aku_utils.common import (
    to_list,
    is_iter
)

KEYS_ORDER = ['name', 't', 'c', 'a', 'l']
TRANSFORMS_ORDER = ['lag', 'mean', 'expmean', 'delta']

keys_order_dict = {key : i for i, key in enumerate(KEYS_ORDER)}
transforms_order_dict = {key : i for i, key in enumerate(TRANSFORMS_ORDER)}


def get_name(cf : Dict):
    '''
    Adds a name key to the config, which is used as the column name

    Special
    ---
    l1, l2: `l1 2 l3 3` -> `2 - 3`
    hidden - contains a value or a dict that will not be included in the name
    '''
    cf = cf.copy()

    def to_str(v):
        '''iterable to string'''
        if is_iter(v):
            return ' '.join([str(i) for i in v])
        return v

    main = [cf.pop('t'), cf.pop('c')]

    trues = [k for k, v in cf.items() if v is True]
    for k in trues:
        del cf[k]

    params = []

    # start of special symbol parsing
    cf.pop('hidden', None)

    # lag 1 and lag 2 for delta
    if 'l1' in cf and 'l2' in cf:
        params += [f"{cf.pop('l1')}-{cf.pop('l2')}"]

    # end of special symbol parsing
    params += [f'{k} {to_str(v)}' for k, v in cf.items()]

    name = ' '.join(
        main + trues + params
    )
    return name


def add_name(cf: Dict):
    cf['name'] = get_name(cf)
    return cf


def flatten(cfs : Union[Dict, List[Dict]]):
    '''
    Flattens a (list of) config(s). See Usage

    Usage
    ---
    ```python
    flatten({'t' : 'lag', 'c' : 'target', 'l' : [0, 1, 2]})
    >>> [{'t': 'lag', 'c': 'target', 'l': 0},
    {'t': 'lag', 'c': 'target', 'l': 1},
    {'t': 'lag', 'c': 'target', 'l': 2}]
    ```
    '''
    if isinstance(cfs, dict):
        cfs = [cfs]

    flattened_cfs = []
    for cf in cfs:
        res = {k : to_list(v) for k, v in cf.items()}

        res = [
            # zips every values combination with original keys
            # and turns it into a dict
            dict(zip(res.keys(), values_set))
            for values_set in product(*res.values())  # produces every values combination
        ]
        flattened_cfs.extend(res)
    return flattened_cfs


def flatten_delta_chain(cfs : Union[Dict, List[Dict]]):
    '''
    Flattens a (list of) config(s). See Usage
    Specifically for delta transform, producing "chained" lags (0-1, 1-2 etc)

    Arguments
    ---
        cfs: a (list of) config(s). Must contain `l` keyword which is
        used for new column generation

    Usage
    ---
    ```python
    config = [
        {'t' : 'delta', 'c' : 'target', 'l' : range(4)},
        {'t' : 'delta', 'c' : 'event_a', 'l' : range(3)},
    ]
    >>> flatten_delta_chain(config)
    [{'t': 'delta', 'c': 'target', 'l1': 0, 'l2': 1},
    {'t': 'delta', 'c': 'target', 'l1': 1, 'l2': 2},
    {'t': 'delta', 'c': 'target', 'l1': 2, 'l2': 3},
    {'t': 'delta', 'c': 'event_a', 'l1': 0, 'l2': 1},
    {'t': 'delta', 'c': 'event_a', 'l1': 1, 'l2': 2}]
    ```
    '''
    if isinstance(cfs, dict):
        cfs = [cfs]

    flattened_cfs = []
    for cf in cfs:
        curr_cf = cf.copy()

        del curr_cf['l']

        new_cf_list = [
            {**curr_cf, 'l1' : l1, 'l2' : l1 + 1}
            for l1 in cf['l'][:-1]
        ]
        flattened_cfs.extend(new_cf_list)
    return flattened_cfs


def flatten_delta_base(cfs : Union[Dict, List[Dict]]):
    '''
    Flattens a (list of) config(s). See Usage
    Specifically for delta transform, producing "base" lags (0-1, 0-2 etc)

    Arguments
    ---
        cfs: a (list of) config(s). Must contain `l` keyword which is
        used for new column generation

    Usage
    ---
    ```python
    config = [
        {'t' : 'delta', 'c' : 'target', 'l' : range(4)},
        {'t' : 'delta', 'c' : 'event_a', 'l' : range(3)},
    ]
    >>> flatten_delta_base(config)
    [{'t': 'delta', 'c': 'target', 'l1': 0, 'l2': 1},
    {'t': 'delta', 'c': 'target', 'l1': 0, 'l2': 2},
    {'t': 'delta', 'c': 'target', 'l1': 0, 'l2': 3},
    {'t': 'delta', 'c': 'event_a', 'l1': 0, 'l2': 1},
    {'t': 'delta', 'c': 'event_a', 'l1': 0, 'l2': 2}]
    ```
    '''
    if isinstance(cfs, dict):
        cfs = [cfs]

    flattened_cfs = []
    for cf in cfs:
        curr_cf = cf.copy()

        del curr_cf['l']

        new_cf_list = [
            {**curr_cf, 'l1' : cf['l'][0], 'l2' : l2}
            for l2 in cf['l'][1:]
        ]
        flattened_cfs.extend(new_cf_list)
    return flattened_cfs


def pick_out_duplicates(cfs : List[Dict]):
    '''picks out duplicates, storing them in a separate list
    only works on dicts'''
    seen = set()
    duplicates = set()

    for d in cfs:
        # this step may scrumble key order
        # intentionally left out bc
        # we sort keys later anyway

        # fails if any part of a tuple is a list
        # so we must enforce no lists as parameters
        dict_tuple = tuple(sorted(d.items()))

        if dict_tuple in seen:
            duplicates.add(dict_tuple)

        seen.add(dict_tuple)

    uniques = [dict(t) for t in seen]
    duplicates = [dict(t) for t in duplicates]
    return uniques, duplicates


def postprocess(cf):
    '''add special stuff like column names to find for delta transform

    inplace operation, so it modifies the original dictionary'''
    if cf['t'] == 'delta':
        base = {'t' : 'lag', 'c' : cf['c']}
        first = {**base, 'l' : cf['l1']}
        second = {**base, 'l' : cf['l2']}
        cf.setdefault('hidden', {}).update({
            'first_name' : get_name(first),
            'second_name' : get_name(second),
        })
    return cf


def order_keys(cf : Dict) -> Dict:
    '''
    sorts config's keys (parameters) in accordance to KEYS_ORDER
    '''
    res = sorted(
        cf.items(),
        key = lambda key_value_tuple:
        keys_order_dict.get(key_value_tuple[0], 999)
    )
    res = dict(tuple(res))
    return res


def validate(cf):
    '''
    forces iterables into tuples

    TODO add field validation, eg lag mustn't have `window`, and expmean must have `alpha`
    '''
    def iters_to_tuple(v):
        if is_iter(v):
            return tuple(v)

        return v

    # forcing lists into tuples
    cf = {k : iters_to_tuple(v) for k, v in cf.items()}
    return cf


def finalize(cfs : List[Dict[str, Any]]):
    '''
    Finalize configs, which includes:
    * adding names to configs
    * removing duplicates
    * sorting keys in config and configs themselves
    '''
    if isinstance(cfs, dict):
        cfs = [cfs]

    cfs = [validate(cf) for cf in cfs]

    # remove duplicates, pass through unique configs
    cfs, duplicates = pick_out_duplicates(cfs)

    if duplicates:
        warnings.warn(f"duplicate configs found: {duplicates}")

    # cfs = [postprocess(cf) for cf in cfs]

    # add names keywords to configs
    cfs = [add_name(cf) for cf in cfs]

    # order keys in each configs in a specific order
    cfs = [order_keys(cf) for cf in cfs]

    # order configs by 1) TRANSFORMS_ORDER
    # 2) column name (alphabetically)
    # 3) lag parameter
    cfs = sorted(
        cfs,
        key = lambda cf: (
            transforms_order_dict.get(cf['t'], 999),
            cf['c'],
            cf.get('l', 999)
        )
    )
    return cfs
