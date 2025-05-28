from aku_utils import this_week, today

datetime_functions = {
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
    'week' : lambda srs: srs.dt.strftime('%Y%V').astype('int'),
    'relative_date' : lambda srs: (srs - today).dt.days,
    'relative_week' : lambda srs: (
        srs.dt.strftime('%Y%V').astype('int')
        - this_week
    )
}

# can be used by:
# df.assign(**{
#     name : ak.calc.datetime_functions[name](df['dt'])
#     for name in ak.calc.datetime_base_features
# })
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
