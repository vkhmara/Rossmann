import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from datetime import datetime, timedelta

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
        self.prophet = Prophet()

    def fit(self, X, y):
        start_date = datetime(2013, 1, 1)
        df = pd.DataFrame({
            'ds': X['ordered_day'].apply(lambda d: start_date + timedelta(days=d)),
            'y': y,
            'Store': X['Store'],
        })
        self.prophet.fit()

    def predict(self, X):
        return super().predict(X)