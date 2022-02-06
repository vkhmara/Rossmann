import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from rossmann.consts import DATE_COL

from rossmann.regressor import Regressor
from rossmann.transformer import Transformer

class Experiment:
    TRAIN = 0
    VAL = 1
    TEST = 2

    @staticmethod
    def metric(y_true, y_pred):
        y_t = y_true[np.abs(y_true) > 1e-5]
        y_p = y_pred[np.abs(y_true) > 1e-5]
        return np.sqrt(np.mean((1 - y_p / y_t)**2))

    def __init__(self, df_filename, store_filename, regressor: Regressor, transformer: Transformer,
                 n_folds=5, window_part=0.6, val_part=48, test_part=48):

        # just saving info
        self.n_folds = n_folds
        self.window_part = window_part
        self.val_part = val_part
        self.test_part = test_part
        self.regressor = regressor

        # read the dataset and store tables
        self.df = pd.read_csv(df_filename)
        self.store = pd.read_csv(store_filename)

        # creating DATE_COL column for slicing the dataset
        self.df[DATE_COL] = self.df['Date'].apply(
            lambda date: datetime.datetime.strptime(date, '%Y-%m-%d'))
        start_date = self.df[DATE_COL].min()
        self.df['ordered_day'] = self.df[DATE_COL].apply(
            lambda date: (date - start_date).days)

        # calculating the window length
        all_time_values = self.df['ordered_day'].unique()
        min_day, max_day = min(all_time_values), max(all_time_values)
        n = max_day - min_day
        window_len = int(n * self.window_part)

        # if test_part is the part of dataset to be tested
        # then it will become the length of the test part
        if isinstance(self.test_part, float):
            self.test_part = int(window_len * self.test_part)

        # analogically the previous block
        if isinstance(self.val_part, float):
            self.val_part = int(window_len * self.val_part)

        # the dataframes for all slices will be saved
        self.train_dfs = []
        self.val_dfs = []
        self.test_dfs = []

        # the total information about the datasets of
        # each slice will be saved
        self.train_date_info = []
        self.val_date_info = []
        self.test_date_info = []

        for fold in range(self.n_folds):
            # calculation the first and the last day of fold
            end_day = min_day + window_len - 1 + \
                (fold * (n - window_len)) // (self.n_folds - 1)
            start_day = end_day - window_len + 1

            # choosing only necessary data from table and
            # extracting target from this
            mask = (start_day <= self.df['ordered_day']) & (
                self.df['ordered_day'] <= end_day)
            df = self.df[mask]
            X, y = df.drop(columns='Sales'), df['Sales']

            # right borders of train and val datasets
            val_end_day = int(end_day - self.test_part)
            train_end_day = int(val_end_day - self.val_part)

            # masks for extracting the datasets
            train_mask = (start_day <= X['ordered_day']) & (
                X['ordered_day'] <= train_end_day)
            val_mask = (train_end_day + 1 <=
                        X['ordered_day']) & (X['ordered_day'] <= val_end_day)
            test_mask = (val_end_day + 1 <=
                         X['ordered_day']) & (X['ordered_day'] <= end_day)

            # extracting the train, val and test dataset
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # the information about each dataset is saved
            self.train_date_info.append(
                (X_train[DATE_COL].min(),
                 X_train[DATE_COL].max(),
                 (X_train[DATE_COL].max() -
                  X_train[DATE_COL].min()).days + 1
                 ))
            self.val_date_info.append(
                (X_val[DATE_COL].min(),
                 X_val[DATE_COL].max(),
                 (X_val[DATE_COL].max() - X_val[DATE_COL].min()).days + 1))
            self.test_date_info.append(
                (X_test[DATE_COL].min(),
                 X_test[DATE_COL].max(),
                 (X_test[DATE_COL].max() - X_test[DATE_COL].min()).days + 1))

            # fit transformer and transform the datasets
            transformer.fit(X_train, y_train)
            self.train_dfs.append(transformer.transform(X_train, y_train))
            self.val_dfs.append(transformer.transform(X_val, y_val))
            self.test_dfs.append(transformer.transform(X_test, y_test))

    def run(self, models_folder, mode='r'):

        # scores of each fold are saved
        train_scores = []
        val_scores = []
        test_scores = []

        # scores of each fold and each horizont are saved
        val_horiz_scores = []
        test_horiz_scores = []

        # these arrays are needed just for loop iteration
        scores = [train_scores, val_scores, test_scores]
        horiz_scores = [None, val_horiz_scores, test_horiz_scores]
        dataset_types = [Experiment.TRAIN, Experiment.VAL, Experiment.TEST]

        # the predictions of each fold.
        # They contain true, prediction, DATE_COL and Store columns
        self.predictions = []

        for fold in range(self.n_folds):
            # scores on val and test datasets depending on horizont
            val_horiz_scores.append([])
            test_horiz_scores.append([])

            # just get the necessary datasets from arrays
            X_train, y_train = self.train_dfs[fold]
            X_val, y_val = self.val_dfs[fold]
            X_test, y_test = self.test_dfs[fold]

            # read the fitted model from file
            if mode == 'r':
                with open(models_folder + f'model_{fold}.pickle', 'rb') as f:
                    self.regressor = pickle.load(f)
            # or fit it and write to file
            else:
                self.regressor.fit(
                    X_train.drop(columns=DATE_COL),
                    y_train,
                    X_val.drop(columns=DATE_COL),
                    y_val)

                if mode == 'w':
                    with open(models_folder + f'model_{fold}.pickle', 'wb') as f:
                        pickle.dump(self.regressor, f)

            # combining all datasets for predicting on all data
            # and further processing the results
            all_X = pd.concat([X_train, X_val, X_test],
                              axis=0, ignore_index=True)
            all_y = pd.concat([y_train, y_val, y_test],
                              axis=0, ignore_index=True)

            # predict all and save it
            prediction = self.regressor.predict(
                all_X.drop(columns=DATE_COL))
            prediction = pd.concat([all_X[['Store', DATE_COL]],
                                    pd.Series(all_y, name='true'),
                                    pd.Series(
                prediction, name='prediction'),
                pd.Series(
                [Experiment.TRAIN] * len(X_train) +
                [Experiment.VAL] * len(X_val) +
                [Experiment.TEST] * len(X_test),
                name='dataset_type'
            )],
                axis=1)
            self.predictions.append(prediction)

            # processing the results of predicting
            # the train, val and test datasets
            for score, dataset_type, horiz_score in zip(scores, dataset_types, horiz_scores):

                # extracting the data of the dataset_type
                prediction_of_dataset_type = prediction.loc[prediction['dataset_type'] == dataset_type]
                score.append(Experiment.metric(prediction_of_dataset_type['true'],
                                               prediction_of_dataset_type['prediction']))

                if dataset_type == Experiment.TRAIN:
                    continue

                # for val and test datasets the scores depending
                # on horizont are calculated
                ds = prediction_of_dataset_type[DATE_COL]
                all_dates = np.sort(ds.unique())
                for date in all_dates:
                    prediction_of_dataset_type_horiz = prediction_of_dataset_type[
                        prediction_of_dataset_type[DATE_COL] == date]
                    horiz_score[fold].append(
                        Experiment.metric(prediction_of_dataset_type_horiz['true'],
                                          prediction_of_dataset_type_horiz['prediction']))

        # cast the datasets and horizont scores information to dataframes
        train_bounds = pd.DataFrame(self.train_date_info, columns=[
                                    'train_start_date', 'train_end_date', 'train_days'])
        val_bounds = pd.DataFrame(self.val_date_info, columns=[
                                  'val_start_date', 'val_end_date', 'val_days'])
        test_bounds = pd.DataFrame(self.test_date_info, columns=[
                                   'test_start_date', 'test_end_date', 'test_days'])
        test_horiz_scores = pd.DataFrame(test_horiz_scores,
                                         columns=[f'test_score_{i+1}' for i in range(self.test_part)])

        # common results about experiment
        self.results = pd.concat([
            pd.DataFrame({
                'fold': range(self.n_folds),
                'train_score': train_scores,
                'val_score': val_scores,
                'test_score': test_scores,
            }),
            train_bounds, val_bounds, test_bounds,
            test_horiz_scores
        ], axis=1)

        # reducing the memory size
        del self.train_dfs
        del self.val_dfs
        del self.test_dfs

        self.view_horizont_errors()
        self.view_dataset_slicing()
        # self.view_prediction_results()

        return self.results

    def view_horizont_errors(self):
        for fold in range(self.n_folds):
            fold_info = self.results[self.results['fold'] == fold]

            test_horizonts = np.arange(1, self.test_part + 1)
            test_horiz_scores = fold_info[[
                f'test_score_{i}' for i in test_horizonts]].values[0]

            plt.figure(figsize=(12, 7))
            plt.title(f'errors of horizonts, fold={fold}')
            plt.plot(test_horizonts, test_horiz_scores)
            plt.plot(test_horizonts[6:],
                pd.Series(test_horiz_scores).rolling(7).mean().dropna(),
                c='r')
            plt.legend(['test', 'rolling mean'])
            plt.xlabel('horizont')
            plt.ylabel('error')
            plt.grid(axis='y', ls='--')

            plt.show()

    def view_dataset_slicing(self):
        plt.figure(figsize=(12, 7))
        plt.title('dataset slicing')
        for fold in range(self.n_folds):
            plt.plot(self.train_date_info[fold][:2], [fold, fold], color='b')
            plt.plot(self.val_date_info[fold][:2], [fold, fold], color='y')
            plt.plot(self.test_date_info[fold][:2], [fold, fold], color='r')
        plt.legend(['train', 'val', 'test'])
        plt.yticks(range(5), range(5))
        plt.grid(b=True, ls='--')
        plt.xlabel('date')
        plt.ylabel('the number of fold')
        plt.show()

    @staticmethod
    def RMSPE(col):
        return np.sqrt((col**2).sum() / len(col))

    @staticmethod
    def get_cat_errors(error_df, cat):
        return error_df.groupby(cat, dropna=False).agg(
            {'rspe': Experiment.RMSPE}).rename(columns={'rspe': 'rmspe'})

    @staticmethod
    def plot_cat_rmspe(error_df, cat, ax):
        errs = Experiment.get_cat_errors(error_df, cat)
        ax.set_title('rmspe for category\n' + cat)
        errs['rmspe'].plot(kind='barh', ax=ax)
        ax.set_xlabel('error')
        ax.set_ylabel(cat)
        ax.grid(axis='x')

    @staticmethod
    def plot_diff_cat_rmspe(error_df1, error_df2, cat, ax):
        errs1 = Experiment.get_cat_errors(error_df1, cat)
        errs2 = Experiment.get_cat_errors(error_df2, cat)
        errs = errs1 - errs2
        ax.set_title('difference rmspe for category\n' + cat)
        errs['rmspe'].plot(kind='barh', ax=ax)
        ax.set_xlabel('difference error')
        ax.set_ylabel(cat)
        ax.grid(axis='x')

    def view_prediction_results(self):
        cats = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
                'Assortment', 'Promo2', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', ]
        self.error_dfs = []
        for fold in range(self.n_folds):
            pred = self.predictions[fold].merge(self.store, on='Store')
            pred['rspe'] = (pred['true'] - pred['prediction']) / pred['true']
            error_df = self.df.merge(pred, on=[DATE_COL, 'Store'])
            error_df = error_df[~error_df.rspe.abs().isin([-np.inf, np.inf])]
            error_df['StateHoliday'].replace(0, '0', inplace=True)
            self.error_dfs.append(error_df)
        width = 30
        for cat in cats:
            height = 0.7 * self.error_dfs[0][cat].nunique()
            fig, axes = plt.subplots(1, self.n_folds, figsize=(width, height))
            for fold, error_df in enumerate(self.error_dfs):
                Experiment.plot_cat_rmspe(error_df, cat, axes[fold])
            plt.show()


def compare_experiments(exps: list, exp_names: list):
    """
    The experiments must be already run
    """
    n_folds = exps[0].n_folds
    assert len(exps) > 1, 'not enough experiments to compare'
    assert all(map(lambda exp: exp.n_folds == n_folds, exps)), 'not all experiments are compatible'

    plt.figure(figsize=(12, 7))
    for exp in exps:
        plt.plot(exp.results['test_score'])
        plt.xlabel('fold')
        plt.ylabel('error')
        plt.title('errors of models in folds')

    plt.legend(exp_names)
    plt.show()
