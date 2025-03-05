'''
Transforms module

Contains a submodule handling configs generation and processing
'''
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Union, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

class Lags(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        prt,
        configs
    ):
        self.prt = prt
        self.configs = configs

    def fit(self, *args):
        return self
    
    def transform(self, df):
        dfat = df.assign(**{
            cf['name'] : self.transform_map[cf['t']](self, df=df, cf=cf)
            for cf in self.configs
        })
        return dfat

    def lag(self, df, cf):
        srs = df.groupby(self.prt)[cf['c']].shift(cf['l'])
        return srs

    # must be after all the transforms
    transform_map = {
        'lag' : lag
    }


class SpecialDays(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        config,
        dt_col = 'dt',
    ):
        '''
        Adds binary features depending on whether datetime column
        values lie within period specified in the config

        Arguments
        ---
            config:
                key: name of the period, will be used as a column name
                value: one (one date) or two (range of two dates) dates of format
                `%m-%d %H:%M` or `%m-%d`
            dt_col:
                name of the datetime column

        Usage
        ---
        ```python
        special_days = {
            'standalone' : '03-08',
            'standalone hour' : '03-08 12:00',
            'range hourly overreach' : ('12-25 16:00', '01-10 12:00'),
            'range overreach' : ('12-25', '01-10'),
            'range hourly' : ('01-20 16:00', '02-10 12:00'),
            'range' : ('01-20', '02-10'),
            'range hourly left' : ('01-20 16:00', '02-10'),
            'range hourly right' : ('01-20', '02-10 12:00'),
        }

        pipeline = Pipeline(steps=[('special_days', SpecialDays(special_days, 'dt'))])
        dfat = pipeline.fit_transform(df)
        ```
        '''
        self.dt_col = dt_col
        self.config = self.parse_config(config)
    

    @staticmethod
    def parse_date(v):
        '''
        Datetime parsing would not work because i need to know if the user specified the hour or not,
        and datetime parsing would just return 0 in any case
        '''
        import re
        pattern = r"^(\d{2})-(\d{2}) ?(\d{1,2})?(?::\d{2})?"

        date_tuple = re.match(pattern, v)

        if date_tuple is None:
            print('invalid format. only `%m-%d %H:%M` or `%m-%d`')
            raise ValueError
        else:
            date_tuple = date_tuple.groups()

        date_tuple = [int(date_part) for date_part in date_tuple if date_part is not None]
        date_dict = dict(zip(['month', 'day', 'hour'], date_tuple))
        return date_dict


    def parse_config(
        self,
        config : Dict[str, Union[str, Tuple[str, str]]]
    ) -> List[Dict[str, Union[Dict, Tuple[Dict, Dict]]]]:
        final_config = []
        for name, values in config.items():
            cf = dict()
            cf['name'] = name

            if isinstance(values, str):
                cf['period_type'] = 'standalone'
                date = self.parse_date(values)
                cf['time_type'] = 'hourly' if date.get('hour') else 'daily'
                cf['periods'] = date
            else:
                cf['period_type'] = 'range'
                start, end = self.parse_date(values[0]), self.parse_date(values[1])

                if start.get('hour') is not None or end.get('hour') is not None:
                    cf['time_type'] = 'hourly'
                    start['hour'] = start.get('hour', 0)
                    end['hour'] = end.get('hour', 23)
                else:
                    cf['time_type'] = 'daily'
                cf['periods'] = (start, end)
            
            final_config.append(cf)
        return final_config

    def fit(self, df, *args):
        self.years = list(range(
            df[self.dt_col].min().year, df[self.dt_col].max().year + 1
        ))

        self.config = [
            self.add_specific_periods(config, years=self.years)
            for config in self.config
        ]

        return self


    def add_specific_periods(
        self,
        cf : Dict[str, Any],
        years : List[int],
    ) -> Dict[str, Any]:
        '''
        Modifies original config
        '''
        dt_format = '%Y-%m-%d'
        if cf['time_type'] == 'hourly':
            dt_format += ' %H:%M'

        if cf['period_type'] == 'range':
            start = cf['periods'][0]
            end = cf['periods'][1]
            next_year_reach = (
                (start['month'], start['day']) >
                (end['month'], end['day'])
            )
            cf['periods'] = [
                (
                    datetime(
                        year=year, month=start['month'], day=start['day'], hour=start.get('hour', 0)
                    ).strftime(dt_format),
                    datetime(
                        year=year+int(next_year_reach), month=end['month'], day=end['day'], hour=end.get('hour', 0)
                    ).strftime(dt_format),
                )
                for year in years
            ]
        else:
            period = cf['periods']
            cf['periods'] = [
                datetime(
                    year=year, month=period['month'], day=period['day'], hour=period.get('hour', 0)
                ).strftime(dt_format)
                for year in years
            ]
        return cf


    def transform(self, df):
        dfat = df.assign(**{
            config['name'] : self.apply_period(df, config)
            for config in self.config
        })
        return dfat

    def apply_period(self, df, cf):
        binary_array : pd.Series = None

        dt_srs = (
            df['dt'] if cf['time_type'] == 'hourly'
            else df['dt'].dt.normalize()
        )

        # period - str or tuple of str
        for period in cf['periods']:
            # this binary array
            tba = (
                dt_srs == period if cf['period_type'] == 'standalone'
                else dt_srs.between(period[0], period[1])
            )

            # combining filters
            binary_array = (
                tba if binary_array is None
                else binary_array | tba
            )
        binary_array = binary_array.astype('int')
        return binary_array