import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


DATA_FILE = 'data_ai_course_2015_2018.csv'


def load_data_for_univariate_regression():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""

    df = pd.read_csv(DATA_FILE)

    X = df.hours_exam.values[:, np.newaxis]
    y = df.points_exam.values
    return X, y


def load_data_for_multivariate_regression():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays."""
    df = pd.read_csv(DATA_FILE)
    X = df.xs(['hours_exam', 'hours_per_week', 'hours_per_week_ai', 'lecture_attendance'], axis=1).values
    y = df.points_exam.values
    return X, y


def plot_2d_data(X, y):
    """Plot the data in 2D space, assuming X is 1-dimensional."""
    plt.figure()
    plt.scatter(X, y)
    plt.xlabel('hours_exam', fontsize=16)
    plt.ylabel('points_exam', fontsize=16)


def plot_predictions(w, c='r', label=None):
    """Plot predictions of the linear model with parameters w."""
    # Assuming there exists a plot of the data,
    # read the axes limits:
    ax = plt.axis()
    # Build the test data as xmin and xmax:
    Xtst = np.array(ax[0:2])[:, np.newaxis]
    # Find the predictions
    ytst = pred_regr_lin(w, Xtst)
    # Plot them
    plt.plot(Xtst, ytst, lw=3, c=c, label=label)
    # Set axes limits as they were before
    plt.axis(ax)
    plt.legend(loc='upper left')


def homogenize(X):
    """Return X with a prepended column of ones."""

    N = np.size(X)
    Xh = np.ones((N, 2))
    Xh[:, :-1] = X
    return Xh


def pred_regr_lin(w, X):
    """Predict the values of y for inputs X, given the linear model parameters w."""
    w = np.mat(w)
    X = np.array(X)
    Xnew = homogenize(X)
    wT = np.transpose(w)
    yhat = Xnew*wT
    return yhat


def compute_err_mse(y, yhat):
    """Return the mean squared error given the true values of y and predictions yhat."""
    y = np.array(y)
    yhat = np.array(yhat)
    ax = 0
    err = (np.square(y - yhat[:,0])).mean(axis=ax)
    if np.shape(yhat)[1] > 1:
        err2 = (np.square(y - yhat[:,1])).mean(axis=ax)
        err = [err,err2]
    return err


def compute_cost_regr_lin(w, X, y):
    """Return the cost of lin. reg. and its gradient at point w, given data X and y."""
    yhat = pred_regr_lin(w, X)
    err = compute_err_mse(y, yhat)
    return err


def fit_regr_lin_by_minimization(X, y):
    """Return the parameters of linear model trained by MSE minimization."""
    # Possibly some code
    def f(w): return compute_cost_regr_lin(w, X, y)

    # Some other code
    w_init = np.array([0, 0])
    result = minimize(f, w_init, method='bfgs')
    w = result.x

    return w


def fit_regr_lin_by_normal_equation(X, y):
    """Return the parameters of linear model using normal equation."""
    X = homogenize(X)
    print(y)
    w = np.ndarray.dot(np.linalg.inv(np.ndarray.dot(np.transpose(X), X)), np.ndarray.dot(np.transpose(X), y))
    return w


def pretty(x, y):
    print(" Variable ", y, ": ", x, " shape: ", np.shape(x))
