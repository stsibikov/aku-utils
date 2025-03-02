'''
Transforms module

Contains a submodule handling configs generation and processing
'''
from sklearn.base import BaseEstimator, TransformerMixin

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