'''
A utils module that contains light weight utils for type wrangling.
Used by other modules
'''

from datetime import datetime

curr_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def to_list(arg):
    if isinstance(arg, str) or not hasattr(arg, '__iter__'):
        return [arg]
    return list(arg)


def is_iter(v):
    return hasattr(v, '__iter__') and not isinstance(v, str)

