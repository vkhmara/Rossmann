import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from datetime import datetime, timedelta
import time

class Regressor:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return 0

    def metric(self, y_true, y_pred):
        y_t = y_true[np.abs(y_true) > 1e-5]
        y_p = y_pred[np.abs(y_true) > 1e-5]
        return np.sqrt(np.mean((1 - y_p / y_t)**2))

    def score(self, X, y):
        return self.metric(y, self.predict(X))


class LightGBMRegressor(Regressor):
    def __init__(self):
        self.lgbm = LGBMRegressor()
    
    def fit(self, X, y):
        self.lgbm.fit(X.values, y)

    def predict(self, X):
        return self.lgbm.predict(X.values)


class ProphetRegressor(Regressor):
    def __init__(self):
        self.start_date = datetime(2013, 1, 1)

    def fit(self, X, y):
        self.stores = X['Store'].unique()
        y = pd.Series(y, name='target')
        X = pd.concat([X[['Store', 'ordered_day']], y], axis=1)
        all_days = pd.Series(np.arange(X['ordered_day'].min(), X['ordered_day'].max() + 1), name='ordered_day')
        self.dfs = {}
        self.prophets = {
            store: Prophet(uncertainty_samples=0) for store in self.stores
        }
        self.eval_times = []
        for store in self.stores:
            start = time.time()
            df = X.loc[X['Store'] == store, ['ordered_day', 'target']]\
                .merge(all_days, how='right', on='ordered_day').fillna(0)
            df['ordered_day'] = df['ordered_day'].apply(lambda d: self.start_date + timedelta(days=d))
            df.rename(columns={
                'ordered_day': 'ds',
                'target': 'y'
            }, inplace=True)
            self.dfs[store] = df
            self.prophets[store].fit(df)
            end = time.time()
            self.eval_times.append(end - start)
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return self


    def predict(self, X):
        X = X[['Store', 'ordered_day']]
        X['ordered_day'] = X['ordered_day'].apply(lambda d: self.start_date + timedelta(days=d))
        prediction = np.zeros(len(X))
        for store in self.stores:
            # start = time.time()
            store_mask = X['Store'] == store
            if not store_mask.any():
                # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
                continue
            df = X.loc[store_mask, ['ordered_day']]
            df.rename(columns={
                'ordered_day': 'ds',
            }, inplace=True)
            prediction[store_mask] = self.prophets[store].predict(df).yhat.values
            # end = time.time()
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return prediction