import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
import time
import pickle

class Regressor:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return 0

    @staticmethod
    def metric(y_true, y_pred):
        y_t = y_true[np.abs(y_true) > 1e-5]
        y_p = y_pred[np.abs(y_true) > 1e-5]
        return np.sqrt(np.mean((1 - y_p / y_t)**2))

    def score(self, X, y):
        return Regressor.metric(y, self.predict(X))
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class LightGBMRegressor(Regressor):
    def __init__(self, lgbm=None):
        if lgbm is None:
            self.lgbm = LGBMRegressor()
        else:
            self.lgbm = lgbm
    
    def fit(self, X, y):
        self.lgbm.fit(X.values, y)

    def predict(self, X):
        return self.lgbm.predict(X.values)

class ProphetRegressor(Regressor):
    def fit(self, X, y):
        stores = X['Store'].unique()
        y = pd.Series(y, name='y')
        X = pd.concat([X[['Date', 'Store']], y], axis=1)\
            .rename(columns={'Date': 'ds'})
        self.prophets = {
            store: Prophet(uncertainty_samples=0) for store in stores
        }
        self.eval_times = []
        for store in stores:
            start = time.time()
            df = X.loc[X['Store'] == store, ['ds', 'y']]
            self.prophets[store].fit(df)
            end = time.time()
            self.eval_times.append(end - start)
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return self

    def predict(self, X):
        X = X[['Store', 'Date']].rename(columns={'Date': 'ds'})
        stores = X['Store'].unique()
        prediction = np.zeros(len(X))
        for store in stores:
            # start = time.time()
            store_mask = X['Store'] == store
            df = X.loc[store_mask, ['ds']]
            prediction[store_mask] = self.prophets[store].predict(df).yhat.values
            # end = time.time()
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return prediction