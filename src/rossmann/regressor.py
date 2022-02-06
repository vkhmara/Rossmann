import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
import time
import pickle

from rossmann.consts import DATE_COL, PROPHET_DATE_COL

class Regressor:
    def fit(self, X_train, y_train, X_val, y_val):
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
            self.lgbm = LGBMRegressor(n_estimators=1500)
        else:
            self.lgbm = lgbm

    def fit(self, X_train, y_train, X_val, y_val, early_stopping=True, early_stopping_rounds=100):
        if early_stopping:
            self.lgbm.fit(X_train.values, y_train, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=early_stopping_rounds)
        else:
            self.lgbm.fit(X_train.values, y_train)

    def predict(self, X):
        return self.lgbm.predict(X.values)

class ProphetRegressor(Regressor):
    def fit(self, X_train, y_train, X_val, y_val):
        val_time_len = X_val[PROPHET_DATE_COL].nunique()
        train_dates = np.sort(X_train[PROPHET_DATE_COL].unique())[val_time_len:]
        train_mask = X_train[PROPHET_DATE_COL].isin(train_dates)
        X_train, y_train = X_train[train_mask], y_train[train_mask]

        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])

        stores = X['Store'].unique()
        y = pd.Series(y, name='y')
        X = pd.concat([X[[PROPHET_DATE_COL, 'Store']], y], axis=1)
        self.prophets = {
            store: Prophet(uncertainty_samples=0) for store in stores
        }
        self.eval_times = []
        for store in stores:
            start = time.time()
            df = X.loc[X['Store'] == store, [PROPHET_DATE_COL, 'y']]
            self.prophets[store].fit(df)
            end = time.time()
            self.eval_times.append(end - start)
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return self

    def predict(self, X):
        X = X[['Store', PROPHET_DATE_COL]]
        stores = X['Store'].unique()
        prediction = np.zeros(len(X))
        for store in stores:
            # start = time.time()
            store_mask = X['Store'] == store
            df = X.loc[store_mask, [PROPHET_DATE_COL]]
            prediction[store_mask] = self.prophets[store].predict(
                df).yhat.values
            # end = time.time()
            # print(f'Store: {store}\nEvaluation time: {(end - start):.2f} s')
        return prediction
