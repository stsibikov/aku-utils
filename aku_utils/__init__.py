'''
Convention
---
Short versions of variables are not exposed to user unless the method is simple and
no one will provide it as a keyword argument

config / cf - singular, even if config is a list
value / v
dict / d
curr_... - current ...
group - pandas groupby columns
'''
from aku_utils.common import *

from aku_utils import (
    gen,
    transforms,
    calc
)
