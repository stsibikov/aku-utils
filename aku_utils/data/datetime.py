from aku_utils import today
import pandas as pd


def _get_week(date : pd.Series):
    iso = date.dt.isocalendar()[['year', 'week']]
    # attempts to set to UInt32, which results in silent bugs
    return (100 * iso['year'] + iso['week']).astype('int')


datetime_features = {
    'year' : lambda srs: srs.dt.year,
    'month' : lambda srs: srs.dt.month,
    'day' : lambda srs: srs.dt.day,
    'hour' : lambda srs: srs.dt.hour,
    'minute' : lambda srs: srs.dt.minute,
    'second' : lambda srs: srs.dt.second,
    'day_of_week' : lambda srs: srs.dt.dayofweek,  # Monday=0, Sunday=6
    'quarter' : lambda srs: srs.dt.quarter,
    'is_month_start' : lambda srs: srs.dt.is_month_start,
    'is_month_end' : lambda srs: srs.dt.is_month_end,
    'is_quarter_start' : lambda srs: srs.dt.is_quarter_start,
    'is_quarter_end' : lambda srs: srs.dt.is_quarter_end,
    'is_year_start' : lambda srs: srs.dt.is_year_start,
    'is_year_end' : lambda srs: srs.dt.is_year_end,
    'day_of_year' : lambda srs: srs.dt.dayofyear,
    'week_of_year' : lambda srs: srs.dt.isocalendar().week.astype('int'),
    'timestamp' : lambda srs: srs.astype('int64') // 10**9,
    'date' : lambda srs: srs.dt.date,
    'time' : lambda srs: srs.dt.time,
    'is_leap_year' : lambda srs: srs.dt.is_leap_year,
    'week' : _get_week,
    'relative_date' : lambda srs: (srs - pd.Timestamp(today)).dt.days,
    # 'relative_week' : lambda srs: (srs - pd.Timestamp(today)).dt.days // 7,
}

datetime_base_features = [
    'year',
    'month',
    'day',
    'hour',
    'day_of_week',
    'day_of_year',
    'week_of_year',
    'timestamp',
    'week',
    'relative_date',
    'relative_week'
]
