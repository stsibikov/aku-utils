'''
A utils module that contains light weight utils for type wrangling.
Used by other modules
'''

from datetime import datetime

curr_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def is_iter(v):
    '''check if value is an iterable with string not being counted as an iterable'''
    return hasattr(v, '__iter__') and not isinstance(v, str)


def to_list(v):
    if not is_iter(v):
        return [v]
    return list(v)
