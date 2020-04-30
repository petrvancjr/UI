import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

DATA_FILE = 'data_ai_course_2015_2018.csv'


def load_mpg_as_dataframe():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = pd.read_csv('auto-mpg.csv', skipinitialspace=True,
                     names=['mpg', 'cyl', 'disp', 'hp', 'wgt', 'acc', 'year', 'orig', 'name'])
    return df


def load_data_for_univariate_regression():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = load_mpg_as_dataframe()
    X = df.disp.values[:, np.newaxis]
    y = df.hp.values
    return X, y


def load_data_for_multivariate_regression():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = load_mpg_as_dataframe()
    X = df.xs(['mpg', 'cyl', 'disp', 'wgt', 'acc'], axis=1).values
    y = df.hp.values
    return X, y


# noinspection PyPep8Naming
def load_data_2d_classification(columns=None):
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays.
    """
    if not columns:
        columns = ['disp', 'hp']
    df = load_mpg_as_dataframe()
    X = df.xs(columns, axis=1).values
    y = df.orig.values
    y[y > 1] = 0
    return X, y


def plot_xy_data(x, y, xlbl='', ylbl=''):
    """Create plot of x,y pairs."""
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)


def plot_1d_regr_model(model, c='b'):
    """Display predictions of the model in the x-range of an existing figure."""
    # Get the limits of axes in the current graph
    ax = plt.axis()
    # Build a set of points for which we'd like to display the predictions
    Xtst = np.linspace(ax[0], ax[1], 101)
    Xtst = Xtst[:, np.newaxis]
    # Compute the predictions
    # print("model",model)
    ytst = model.predict(Xtst)
    # print("predicted",ytst)


    # Plot them
    plt.plot(Xtst, ytst, lw=3, c=c)
    plt.axis(ax)




def plot_xy_classified_data(x, y, c, xlbl='', ylbl='', colors=None):
    """Create plot of x,y pairs with classes c indicated by different colors."""

    colors = colors if colors is not None else ['b', 'r', 'g', 'y', 'c', 'm']
    unique_classes = set(c)
    for k in unique_classes:
        plt.scatter(x[c == k], y[c == k], c=colors[k], s=36)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)


def plot_2d_class_model(model):
    """Plot the predictions and decision boundary of the model.

    Assumes that a plot of data points already exists.
    """
    ax = plt.axis()
    x1 = np.linspace(ax[0], ax[1], num=101)
    x2 = np.linspace(ax[2], ax[3], num=101)
    mx1, mx2 = np.meshgrid(x1, x2)
    sh = mx1.shape
    vx1 = np.reshape(mx1, (-1, 1))
    vx2 = np.reshape(mx2, (-1, 1))
    vx = np.hstack((vx1, vx2))
    vyhat = model.predict(vx)
    myhat = np.reshape(vyhat, sh)
    plt.contourf(mx1, mx2, myhat, cmap=plt.cm.cool, alpha=0.3)


def compute_err_mse(y, yhat):
    """Compute the mean squared error from the predictions and true values."""
    diffs = y - yhat
    total_err = diffs @ diffs.T / y.shape[0]
    return total_err


def compute_err_01(y, yhat):
    """Compute the zero-one error, i.e. the mis-classification rate."""
    return metrics.zero_one_loss(y, yhat)


def compute_model_error(model, X, y, err_func, pos=True):
    """Computen the error of the model using the give data and error function."""

    r = np.shape(X)[1]
    #print("model.coef_[0:r]",model.coef_[0:r])
    if pos:
        yhat = np.polyval(model.coef_[0][0:r], X)
    else:
        yhat = np.polyval(model.coef_[0:r], X)
    err = np.square(y - yhat[:,0]).sum() / X.shape[0]
    #err = err_func(y, yhat[:,0])

    return err

def homogenize(X):
    """Return X with a prepended column of ones."""

    N = np.size(X)
    Xh = np.ones((N, 2))
    Xh[:, :-1] = X
    return Xh

def my_error(X,y,i,w):
    xx = np.linspace(X.min(), X.max(), 1000)
    yy = np.interp(xx, X, y)

    y_pred = np.polyval(w, xx)

    return np.square(yy - y_pred).sum() / X.shape[0]

def pol_y(x, w):
    y = 0; power = 0;
    for i in w:
        y += i*(x**power);
        power += 1;
    return y