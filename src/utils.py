#!/usr/bin/env python
# -*- coding:utf-8 -*-

import gzip
import shutil

import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from umap import UMAP


def gunzip(gzip_file, output_file, block_size=65536):
    with gzip.open(gzip_file, 'rb') as s_file, \
            open(output_file, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


def scatter_plot(plot_df, **kwargs):
    return sns.scatterplot(
        data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], **kwargs)


def handle_extra_columns(func):
    def wrapper(df, features=None, n_components=2, **kwargs):

        other_columns = None
        if features is not None:
            other_columns = df.columns.difference(features)

        features = slice(None) if features is None else features

        reduced_df = func(df, features, n_components, **kwargs)

        if other_columns is not None:
            reduced_df = reduced_df.join(df[other_columns], how='inner')

        return reduced_df

    return wrapper


@handle_extra_columns
def create_pca_df(df, features=None, n_components=2, **kwargs):

    reducer = PCA(n_components=n_components, **kwargs)
    reducer.fit(df[features])
    reduced_df = pd.DataFrame(
        data=reducer.transform(df[features]),
        index=df.index,
        columns=[f'PC{i + 1} ({v * 100:.1f}%)' for i, v in
                 enumerate(reducer.explained_variance_ratio_)])

    return reduced_df


@handle_extra_columns
def create_umap_df(df, features=None, n_components=2, **kwargs):

    reducer = UMAP(n_components=n_components, **kwargs)
    reducer.fit(df[features])
    reduced_df = pd.DataFrame(
        data=reducer.transform(df[features]),
        index=df.index,
        columns=[f'UMAP{i + 1}' for i in range(n_components)])

    return reduced_df
