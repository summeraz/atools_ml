import random

import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from scipy.stats import gaussian_kde, normaltest


def dimensionality_reduction(df, features, filter_missing=True,
                             filter_var=True, filter_corr=True,
                             filter_nonnorm=False, filter_multimodal=False,
                             missing_threshold=0.4, var_threshold=0.03,
                             corr_threshold=0.9, norm_threshold=0.05,
                             mm_points=25):
    """Reduce the number of feature variables in a dataframe

    Parameters
    ----------
    mm_points : int, optional, default=25
        The number of points used in creating a smoothed histogram (from KDE).
        Lower numbers correspond to increased smoothing.
    """

    df_red = df[features]
    columns_to_recover = df.columns.difference(df_red.columns)

    # Arrange columns in alphabetical order for reproducibility
    df_red = df_red.reindex(sorted(df_red.columns), axis=1)

    '''
    --------------------------------------------
    Remove columns where all values are the same
    --------------------------------------------
    '''
    cols = list(df_red)
    nunique = df_red.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_red.drop(cols_to_drop, axis=1, inplace=True)

    '''
    --------------------
    Missing values ratio
    --------------------
    '''
    if filter_missing:
        valid_columns = df_red.columns[df_red.isnull().mean() < missing_threshold]
        df_red = df_red[valid_columns]

    '''
    -------------------
    Low variance filter
    -------------------
    '''
    if filter_var:
        df_red_norm = (df_red - df_red.mean()) / (df_red.max() - df_red.min())
        to_drop = df_red_norm.var()[df_red_norm.var() < var_threshold].index.values
        df_red.drop(to_drop, axis=1, inplace=True)

    '''
    ----------------------------
    Multimodal distribution test
    ----------------------------
    '''
    if filter_multimodal:
        to_drop = []
        for colname, colvals in df_red.items():
            kernel = gaussian_kde(colvals)
            col0, col1 = min(colvals), max(colvals)
            col_range = col1 - col0
            point_range = np.linspace(col0 - 0.25 * col_range,
                                      col1 + 0.25 * col_range, mm_points)
            colsmooth = kernel.evaluate(point_range)
            if len(argrelmax(colsmooth, order=1)[0]) != 1:
                to_drop.append(colname)
        df_red.drop(to_drop, axis=1, inplace=True)

    '''
    ------------------------
    Normal distribution test
    ------------------------
    '''
    if filter_nonnorm:
        colfilter = df_red.apply(
            lambda vals: normaltest(vals).pvalue < norm_threshold)
        to_drop = [colname for colname, colval in colfilter.items() if colval]
        df_red.drop(to_drop, axis=1, inplace=True)

    '''
    -----------------------
    High correlation filter
    -----------------------
    '''
    if filter_corr:
        df_corr = df_red.corr().abs()
        upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns
                          if any(upper[column] > corr_threshold)]
        df_red.drop(to_drop, axis=1, inplace=True)

    '''
    --------------
    Recover column
    --------------
    '''
    for col in columns_to_recover:
        df_red[col] = df[col]

    return df_red


def train_test_split(df, target, drop=None, by=None, test_size=0.25,
                     random_state=12):
    """
    NOTE: This currently assumes all groups within the dataframe grouped
          by `by` are the same size.
    """
    np.random.seed(random_state)

    if by:
        grouped = df.groupby(by)
        groups = grouped.groups
        keys = [key for key in groups]
        np.random.shuffle(keys)
        n_test = round(len(groups) * test_size)
        test_keys = keys[:n_test]
        train_keys = keys[n_test:]
        test_idx = np.array([np.array(groups[key]) for key in test_keys])
        test_idx = np.concatenate(test_idx).ravel()
        train_idx = np.array([np.array(groups[key]) for key in train_keys])
        train_idx = np.concatenate(train_idx).ravel()
        test = df.drop(drop, axis=1).ix[test_idx]
        train = df.drop(drop, axis=1).ix[train_idx]
    else:
        n_test = round(len(df) * test_size)
        test = df.drop(drop, axis=1).sample(n_test, random_state=random_state)
        train = df.drop(drop, axis=1).loc[~df.index.isin(test.index)]

    return train.drop(target, axis=1), test.drop(target, axis=1), train[target], test[target]
