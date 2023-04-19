#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import joblib
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from dl_omics import create_l1000_df
from utils import create_umap_df, scatter_plot

matplotlib.use('Agg')
sns.set(style='whitegrid')
sns.set_context('paper', font_scale=1.3)

current_file_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(current_file_path, os.path.pardir)


def save_figure(image_path, fig=None):

    if fig is None:
        fig = plt.gcf()

    fig.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0.01,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()


def create_grid_search_plots(grid_search_file, image_folder):

    search = joblib.load(grid_search_file)
    search_result_df = search.trials_dataframe()

    param_name = 'params_selector__k'
    param_display_name = 'k'

    # find configuration with best test score for each k
    best_score_per_k_index = search_result_df.groupby(param_name)['user_attrs_mean_test_score']\
        .idxmax()
    search_result_df = search_result_df.loc[best_score_per_k_index, :]

    # convert results to long format
    param_names = ['params_scaler', 'params_selector__k', 'params_svc__C']
    train_split_names = [c for c in search_result_df.columns if
                         c.startswith('user_attrs_split') and c.endswith('train_score')]
    test_split_names = [c for c in search_result_df.columns if
                        c.startswith('user_attrs_split') and c.endswith('test_score')]
    data = []
    for index, row in search_result_df.iterrows():
        param_values = row[param_names].tolist()
        train_scores = row[train_split_names].tolist()
        test_scores = row[test_split_names].tolist()
        for train_score in train_scores:
            data.append(param_values + ['train', train_score, row.user_attrs_mean_train_score, index])
        for test_score in test_scores:
            data.append(param_values + ['test', test_score, row.user_attrs_mean_test_score, index])

    plot_data = pd.DataFrame(
        data, columns=['scaler', 'k', 'C', 'split', 'MCC', 'mean', 'index'])
    plot_data['scaler'] = plot_data['scaler'].astype(str)
    plot_data = plot_data.rename(columns={'split': 'Split'})

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=plot_data,
        x=param_display_name, y='MCC', hue='Split', hue_order=['train', 'test'], ax=ax
    )

    x_ticks = sorted(plot_data[param_display_name].unique())
    x_ticks = x_ticks[::2]
    ax.set_xticks(x_ticks)

    x = search.best_params_[param_name.replace('params_',  '')]
    y = search.best_score_
    ax.plot(x, y, '*', markersize=15, zorder=-1, alpha=0.8,
            color=ax.lines[1].get_color())

    ax.set_xlim(plot_data[param_display_name].min(), plot_data[param_display_name].max())
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Model performance (MCC)')

    image_path = os.path.join(image_folder, f'figure03_grid_search_{param_display_name}.tiff')
    save_figure(image_path)


def create_data_plots(image_folder):

    df, genes, meta_columns = create_l1000_df()

    target_name = 'DILI'
    df[target_name] = df['dili'].replace({0: 'negative', 1: 'positive'})

    reduced_df = create_umap_df(df, features=genes, densmap=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter_plot(reduced_df, hue='DILI', alpha=0.8, s=70, ax=ax)
    image_path = os.path.join(image_folder, 'figure01_umap_DILI.tiff')
    save_figure(image_path)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    corr_df = df[genes].corr(method='spearman').abs()
    np.fill_diagonal(corr_df.values, np.nan)
    sns.histplot(corr_df.max(axis=1), bins=20, ax=ax1)
    ax1.set_xlabel('Maximum absolute Spearman\'s\ncorrelation between features')

    mi = mutual_info_classif(df[genes], df['dili'])
    sns.histplot(mi, bins=10, ax=ax2)
    ax2.set_xlabel('Mutual information with DILI class')
    ax2.set_ylabel(None)
    image_path = os.path.join(image_folder, 'figure02_corr_mi.tiff')
    save_figure(image_path)


def create_coeff_boxplot(image_folder):

    df, genes, meta_columns = create_l1000_df()

    models_file = os.path.join(base_dir, 'models.pkl')
    models = joblib.load(models_file)
    coefficient_list = []
    for model in models:
        selected_features = model.named_steps['selector'].get_support()
        selected_features = np.array(genes)[selected_features]
        coefficients = np.abs(model.named_steps['svc'].coef_)[0]
        coefficient_list.append(
            pd.DataFrame({'Gene': selected_features, 'Coefficient': coefficients})
        )
    coefficient_df = pd.concat(coefficient_list)

    top_genes = coefficient_df.groupby('Gene').median().sort_values('Coefficient', ascending=False)
    top_genes = top_genes.head(n=5)
    plot_df = coefficient_df.loc[coefficient_df.Gene.isin(top_genes.index), :]

    fig, ax = plt.subplots(figsize=(9, 4))
    image_path = os.path.join(image_folder, 'figure04_coefficients.tiff')
    sns.boxplot(data=plot_df, x='Gene', y='Coefficient', order=top_genes.index, ax=ax)
    ax.set_ylabel('Absolute coefficient')
    save_figure(image_path)


def main():

    image_folder = os.path.join(base_dir, 'figures')
    os.makedirs(image_folder, exist_ok=True)

    grid_search_file = 'grid_search.pkl'
    grid_search_file = os.path.join(base_dir, grid_search_file)
    create_grid_search_plots(grid_search_file, image_folder)

    create_data_plots(image_folder)

    create_coeff_boxplot(image_folder)


if __name__ == '__main__':
    main()
