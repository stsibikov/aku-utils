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
        config : List[Dict[str, Any]],
        prt : str | List[str],
    ):
        '''
        Adds lag features

        Parameters
        ---
            prt:
                partition columns
            configs:
                list of dicts specifying how to generate columns
                (see `aku_utils.transforms.configs`)
        '''
        self.prt = prt
        self.config = config

    def fit(self, *args):
        return self

    def transform(self, df):
        dfat = df.assign(**{
            cf['name'] : self.transform_map[cf['t']](self, df=df, cf=cf)
            for cf in self.config
        })
        return dfat

    def lag(self, df, cf):
        srs = df.groupby(self.prt)[cf['c']].shift(cf['l'])
        return srs

    def mean(self, df, cf):
        srs = df.groupby(self.prt)[cf['c']].shift(cf['l']).rolling(cf['w'], min_periods=1).mean()
        return srs

    def expmean(self, df, cf):
        srs = (
            df.groupby(self.prt)[cf['c']].shift(cf['l'])
            .ewm(alpha=cf['a']).mean()
        )
        return srs

    # must be after all the transforms
    transform_map = {
        'lag' : lag,
        'mean' : mean,
        'expmean' : expmean,
    }


class SpecialDays(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        config,
        month = 'month',
        day = 'day',
        hour = 'hour',
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
            month:
                name of column containing months
            day:
                name of column containing days
            hour:
                name of column containing hours
                if data is daily, just provide any num. column

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
        self.config = self.parse_config(config)
        self.month = month
        self.day = day
        self.hour = hour
    

    def parse_date(self, v):
        '''
        Returns
        ---
            tuple with month, day, hour (may not be present) values
        '''
        try:
            date = datetime.strptime(v, '%m-%d %H:%M')
            date_tuple = date.month, date.day, date.hour
        except ValueError as err:
            try:
                date = datetime.strptime(v, '%m-%d')
                date_tuple = date.month, date.day
            except ValueError as err:
                print(f'{self.__class__.__name__}: invalid date format. see docs.')
                raise err
        
        return date_tuple


    def parse_config(
        self,
        config : Dict[str, Union[str, Tuple[str, str]]]
    ) -> List[Dict[str, Union[Dict, Tuple[Dict, Dict]]]]:
        final_config = []
        for name, values in config.items():
            cf = dict()
            cf['name'] = name

            if isinstance(values, str):
                period = self.parse_date(values)

                # if only month and day was provided, we
                # make the day into a range X 00:00 - X 23:00
                if len(period) < 3:
                    cf['period_type'] = 'range'
                    start = (*period, 0)
                    end = (*period, 23)
                    cf['periods'] = (start, end)
                else:
                    cf['period_type'] = 'standalone'
                    cf['period'] = period
            else:
                cf['period_type'] = 'range'
                start, end = self.parse_date(values[0]), self.parse_date(values[1])

                if len(start) < 3:
                    start = (*start, 0)
                if len(end) < 3:
                    end = (*end, 23)
                cf['periods'] = (start, end)

            final_config.append(cf)
        return final_config

    def fit(self, df, *args):
        return self

    def transform(self, df):
        # date tuple col alias
        dtc = f'{self.__class__.__name__} temp'

        df[dtc] = list(zip(df['obj_id'], df['target'], df['hour']))
        self.date_tuple_col = df[dtc]

        df = df.assign(**{
            config['name'] : self.is_in_period(df, config)
            for config in self.config
        })

        df = df.drop(columns=dtc)
        return df
    
    def is_in_period(self, df, cf):
        if cf['period_type'] == 'range':
            start, end = cf['periods']
            if start <= end:
                srs = (start <= self.date_tuple_col) & (self.date_tuple_col <= end)
            # If the range spans into the next year (e.g., Dec -> Jan)
            else:    
                srs = (self.date_tuple_col >= start) | (self.date_tuple_col <= end)
        else:
            srs = self.date_tuple_col == cf['period']
        
        srs = srs.astype('int')
        return srs