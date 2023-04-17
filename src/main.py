#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
from copy import deepcopy

import numpy as np
import joblib
import pandas as pd
from sklearn import metrics, preprocessing, pipeline, svm, feature_selection, model_selection
from optuna import distributions
from optuna.integration import OptunaSearchCV

from dl_omics import create_l1000_df

current_file_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(current_file_path, os.path.pardir)
RANDOM_STATE = 42


def calculate_scores(estimator, X_test, y_test):

    y_test_predicted = estimator.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_test_predicted)
    mcc = metrics.matthews_corrcoef(y_test, y_test_predicted)
    return accuracy, mcc


def main():

    nested_group_cv = False
    max_local_threads = 4
    n_trials = 50

    target_name = 'dili'
    df, genes, meta_columns = create_l1000_df()

    if nested_group_cv:
        df = df.sample(frac=1, random_state=42)
        train_index_num, test_index_num = next(
            model_selection.GroupKFold(n_splits=5).split(
                df, groups=df['compound']))
        train_index = pd.Series(index=df.index, dtype=bool)
        train_index.iloc[train_index_num] = True
        validation_index = pd.Series(index=df.index, dtype=bool)
        validation_index.iloc[test_index_num] = True
    else:
        train_index = df['set'] == 'Training'
        validation_index = df['set'] == 'Test'

    X_train, y_train = df.loc[train_index, genes], df.loc[train_index, target_name]
    X_validation, y_validation = df.loc[validation_index, genes], \
                                 df.loc[validation_index, target_name]

    svc = svm.LinearSVC(max_iter=50_000, random_state=RANDOM_STATE)
    selector = feature_selection.SelectKBest(
        score_func=feature_selection.mutual_info_classif
    )
    estimator = pipeline.Pipeline([
        ('selector', selector),
        ('scaler', preprocessing.StandardScaler()),
        ('svc', svc)
    ])

    scaler_grid = ['passthrough', preprocessing.StandardScaler(), preprocessing.RobustScaler()]
    param_distributions = {
        'selector__k': distributions.IntDistribution(10, 200),
        'scaler': distributions.CategoricalDistribution(scaler_grid),
        'svc__C': distributions.FloatDistribution(0.001, 1., log=True)
    }
    if nested_group_cv:
        inner_cv = model_selection.GroupShuffleSplit(test_size=0.2, n_splits=10, random_state=RANDOM_STATE)
        grid_search_file = 'grid_search_nested_group.pkl'
    else:
        inner_cv = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
        grid_search_file = 'grid_search.pkl'
    grid_search_file = os.path.join(base_dir, grid_search_file)

    mcc_scorer = metrics.make_scorer(metrics.matthews_corrcoef, greater_is_better=True)

    try:
        search = joblib.load(grid_search_file)
    except FileNotFoundError:
        n_jobs = int(os.environ.get('SLURM_CPUS_ON_NODE', max_local_threads))
        print(f'Trying {n_trials} different configurations using {n_jobs} CPUs')
        search = OptunaSearchCV(
            estimator, param_distributions, cv=inner_cv, n_jobs=n_jobs, n_trials=n_trials,
            random_state=RANDOM_STATE, refit=False, return_train_score=True, scoring=mcc_scorer,
            study=None, verbose=2
        )
        search.fit(X_train, y_train, groups=df.loc[train_index, 'compound'])
        joblib.dump(search, grid_search_file, compress=3)

    print(search.best_params_)
    print(search.best_score_)

    estimator.set_params(**search.best_params_)
    estimator.fit(X_train, y_train)
    n_selected_features = sum(estimator.named_steps['selector'].get_support())
    n_features = X_train.shape[1]
    print(f'{n_selected_features} out of {n_features} features used')

    y_validation_predicted = estimator.predict(X_validation)
    accuracy = metrics.accuracy_score(y_validation, y_validation_predicted)
    mcc = metrics.matthews_corrcoef(y_validation, y_validation_predicted)
    print('accuracy:', accuracy)
    print('MCC:', mcc)

    models = []
    for train_index_num, _ in inner_cv.split(df[genes], df[target_name], groups=df['compound']):
        estimator = deepcopy(estimator)
        estimator.set_params(**search.best_params_)
        estimator.fit(df[genes].iloc[train_index_num], df[target_name].iloc[train_index_num])
        models.append(estimator)

    models_file = 'models_nested.pkl' if nested_group_cv else 'models.pkl'
    models_file = os.path.join(base_dir, models_file)
    joblib.dump(models, models_file, compress=3)


if __name__ == '__main__':
    sys.exit(main())
