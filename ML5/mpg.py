"""
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
Module for loading the mpg dataset. Needs 'auto-mpg.csv' data file.
"""


import numpy as np
import pandas as pd


def load_mpg_as_dataframe():
    """Load 'auto-mpg.csv' and return it as dataframe."""
    # Using pandas to import the data; convenient, but a bit overkill
    df = pd.read_csv('auto-mpg.csv', skipinitialspace=True,
                     names=['mpg', 'cyl', 'disp', 'hp', 'wgt', 'acc', 'year', 'orig', 'name'])
    return df


def load_mpg_for_simple_regression(indep, dep):
    """Returns a data set containing independent variable and dependent variable."""
    df = load_mpg_as_dataframe()
    X = df[indep].values[:, np.newaxis]
    y = df[dep].values
    return X, y


def load_mpg_for_1D_classification():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = load_mpg_as_dataframe()
    X = df.hp.values[:, np.newaxis]
    y = df.orig.values
    y[y > 1] = 0
    return X, y


def load_mpg_for_classification(columns=None):
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = load_mpg_as_dataframe()
    if not columns:
        columns = df.columns[:-2]
    X = df.xs(columns,axis=1).values
    y = df.orig.values
    y[y > 1] = 0
    return X, y

# noinspection PyPep8Naming
def load_mpg_for_2D_classification(columns=None):
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays.
    """
    if not columns:
        columns = ['disp', 'hp']
    df = load_mpg_as_dataframe()
    X = df.xs(columns, axis=1).values
    y = df.orig.values
    y[y > 1] = 0
    return X, y


def load_mpg_for_multiple_regression():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = load_mpg_as_dataframe()
    X = df.xs(['mpg', 'cyl', 'disp', 'wgt', 'acc'], axis=1).values
    y = df.hp.values
    return X, y

