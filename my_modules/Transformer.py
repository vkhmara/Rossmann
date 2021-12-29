from sklearn.preprocessing import StandardScaler    
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import datetime
from operator import attrgetter

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, store):
        self.store = store.copy()
        
    def fit(self, X, y):
        self.store = self.store.merge(np.log10(X.groupby('Store').Customers.sum() - 1.2e5), on='Store')
        
        std_scaler = StandardScaler()
        self.store['Encoded_Customers'] = std_scaler.fit_transform(self.store[['Customers']])
        self.store.drop(columns='Customers', inplace=True)
        
        X_copy = X[X.Open == 1].drop(columns='Open')
        X_copy['target'] = y
        std_scaler = StandardScaler()
        self.store = self.store.merge(X_copy.groupby('Store').target.mean(), on='Store').rename(columns={'target': 'sum_target'})
        self.store['Encoded_sum_target'] = std_scaler.fit_transform(self.store[['sum_target']])
        self.store.drop(columns='sum_target', inplace=True)
        
        return self
    
    def transform(self, X, y=None):
        # choose only necessary for modelling features
        X_transformed = X.copy()[
            ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        ]
        
        if not(y is None):
            X_transformed['target'] = y
        
        # drop rows where Open = 0 as for them the target equals 0
        X_transformed = X_transformed[X_transformed.Open == 1]
        X_transformed.drop(columns='Open', inplace=True)
        
        # it is used to form features
        dates = X_transformed.Date.apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d'))
        
        # create the features that equal the number of days, weeks and years respectively
        # since the start of history
        X_transformed['ordered_day'] = (dates - datetime.datetime(2013, 1, 1)).apply(attrgetter('days'))
        X_transformed['ordered_week'] = X_transformed['ordered_day'] // 7
        X_transformed['ordered_month'] = dates.apply(lambda date: date.month + 12 * (date.year - 2013) - 1)
        X_transformed['MonthOfYear'] = X_transformed['ordered_month'] % 12
        X_transformed.drop(columns='Date', inplace=True)

        # encode change the values 'b' and 'c' of StateHoliday feature to 'a' and then encode
        # it with the rule '0' -> 0, 'a' -> 1 (label encoder)
        X_transformed.StateHoliday = X_transformed.StateHoliday.replace(
                                                    to_replace=['0', 'a', 'b', 'c'], value=[0, 1, 1, 1])

        # merging with the store table
        # I decided to leave the Store feature to research the time series of some stores
        X_transformed = X_transformed.merge(self.store, on='Store')
        if y is None:
            return X_transformed
        return X_transformed.drop(columns='target'), X_transformed.target