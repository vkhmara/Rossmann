from my_modules.Transformer import Transformer
import numpy as np

class Experiments:
    def __init__(self, model, n_folds, window_part, val_part, test_part):
        self.n_folds = n_folds
        self.window_part = window_part
        self.val_part = val_part
        self.test_part = test_part
        self.model = model

    def run(self, X, y):
        tr = Transformer()
        X_transformed = tr.transform(X)
        y_copy = y.copy()
        all_time_values = X_transformed['ordered_day'].unique()
        min_day, max_day = min(all_time_values), max(all_time_values)
        n = max_day - min_day
        window_len = int(n * self.window_part)
        
        for fold in range(self.n_folds):
            end_day = min_day + window_len - 1 + (fold * (n - window_len)) // (self.n_folds - 1)
            start_day = end_day - window_len + 1
            
            mask = start_day <= X_transformed['ordered_day'] <= end_day
            X = X_transformed[mask]
            y = y_copy[mask]
            
            val_end_day = end_day - self.test_part * window_len
            train_end_day = val_end_day - self.val_part * window_len

            train_mask = start_day <= X['ordered_day'] <= train_end_day
            val_mask = train_end_day + 1 <= X['ordered_day'] <= val_end_day
            test_mask = val_end_day + 1 <= X['ordered_day'] <= end_day
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            self.model.fit(X_train, y_train)



