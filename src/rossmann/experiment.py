import datetime
import numpy as np
import pandas as pd

class Experiment:
    def __init__(self, df, regressor, transformer, n_folds=5, window_part=0.6, val_part=0.1, test_part=0.1):
        self.n_folds = n_folds
        self.window_part = window_part
        self.val_part = val_part
        self.test_part = test_part
        self.regressor = regressor

        df_copy = df.copy()

        df_copy['ordered_day'] = df['Date'].apply(
            lambda date: (datetime.datetime.strptime(date, '%Y-%m-%d') -\
                datetime.datetime(2013, 1, 1)).days
                )

        all_time_values = df_copy['ordered_day'].unique()
        min_day, max_day = min(all_time_values), max(all_time_values)
        n = max_day - min_day
        window_len = int(n * self.window_part)

        self.train_dfs = []
        self.val_dfs = []
        self.test_dfs = []

        for fold in range(self.n_folds):
            end_day = min_day + window_len - 1 + (fold * (n - window_len)) // (self.n_folds - 1)
            start_day = end_day - window_len + 1
            
            mask = (start_day <= df_copy['ordered_day']) & (df_copy['ordered_day'] <= end_day)
            df = df_copy[mask]
            X, y = df.drop(columns='Sales'), df['Sales']
            
            val_end_day = end_day - self.test_part * window_len
            train_end_day = val_end_day - self.val_part * window_len

            train_mask = (start_day <= X['ordered_day']) & (X['ordered_day'] <= train_end_day)
            val_mask = (train_end_day + 1 <= X['ordered_day']) & (X['ordered_day'] <= val_end_day)
            test_mask = (val_end_day + 1 <= X['ordered_day']) & (X['ordered_day'] <= end_day)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            transformer.fit(X_train, y_train)
            self.train_dfs.append(transformer.transform(X_train, y_train))
            self.val_dfs.append(transformer.transform(X_val, y_val))
            self.test_dfs.append(transformer.transform(X_test, y_test))

    def run(self):

        train_scores = []
        val_scores = []
        test_scores = []

        for fold in range(self.n_folds):
            X_train, y_train = self.train_dfs[fold]
            X_val, y_val = self.val_dfs[fold]
            X_test, y_test = self.test_dfs[fold]
            
            self.regressor.fit(X_train, y_train)
            
            train_scores.append(self.regressor.score(X_train, y_train))
            val_scores.append(self.regressor.score(X_val, y_val))
            test_scores.append(self.regressor.score(X_test, y_test))
        
        return pd.DataFrame({
            'fold': range(self.n_folds),
            'train_score': train_scores,
            'val_score': val_scores,
            'test_score': test_scores,
        })



