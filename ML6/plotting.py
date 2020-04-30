"""
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Module containing plotting functions.
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def plot_xy_data(x, y, c='b', xlbl='x', ylbl='y', s=20):
    """Create plot of x,y pairs."""
    plt.scatter(x, y, c=c, s=s)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)
    _draw()


def plot_xy_classified_data(x, y, c, xlbl='x', ylbl='y', colors=None):
    """Create plot of x,y pairs with classes c indicated by different colors."""
    if colors is None:
        colors = ['b', 'r', 'g', 'y', 'c', 'm']
    unique_classes = set(c)
    for k in unique_classes:
        plt.scatter(x[c == k], y[c == k], c=colors[k], s=36)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)
    plt.grid(True)
    # _draw()


def plot_1D_regr_model(model, c='b'):
    """Display predictions of the model in the x-range of an existing figure."""
    # Get the limits of axes in the current graph
    ax = plt.axis()
    # Build a set of points for which we'd like to display the predictions
    Xtst = np.linspace(ax[0], ax[1], 201)
    Xtst = Xtst[:, np.newaxis]
    # Compute the predictions
    ytst = model.predict(Xtst)
    # Plot them
    p = plt.plot(Xtst, ytst, lw=3, c=c)
    plt.axis(ax)
    # _draw()
    return p


def plot_2D_class_model(model, offset=0):
    """Plot the predictions and decision boundary of the model.

    Assumes that a plot of data points already exists.
    """
    ax = plt.axis()
    x1 = np.linspace(ax[0] - offset, ax[1] + offset, num=101)
    x2 = np.linspace(ax[2] - offset, ax[3] + offset, num=101)
    mx1, mx2 = np.meshgrid(x1, x2)
    sh = mx1.shape
    vx1 = np.reshape(mx1, (-1, 1))
    vx2 = np.reshape(mx2, (-1, 1))
    vx = np.hstack((vx1, vx2))
    vyhat = model.predict(vx)
    myhat = np.reshape(vyhat, sh)
    plt.contourf(mx1, mx2, myhat, lw=3, cmap=plt.cm.cool, alpha=0.3)
    _draw()


def plot_roc(models, Xtr, ytr, Xtst, ytst, nfeat=np.inf):
    """
    Compute and plot ROC curves for given models and data.

    :param models: A list of dicts describing the models.
                Each dictionary should have the following items:
               'clf' - the classifier
               'descr' - str description of classifier
               'color' -  color used for plotting
    :param Xtr: training features
    :param ytr: training labels
    :param Xtst: test features
    :param ytst: test labels
    :param nfeat: number of features to be used in the model
    :return:
    """

    nfeat = Xtr.shape[1] if nfeat > Xtr.shape[1] else nfeat
    Xtr = Xtr[:, 0:nfeat]
    Xtst = Xtst[:, 0:nfeat]

    for m in models:
        keys = m.keys()
        assert 'clf' in keys
        assert 'descr' in keys
        assert 'color' in keys

    for model in models:
        # Process the model information
        clf = model['clf']
        descr = model['descr']
        color = model['color']
        # Fit the classifier to training data
        clf.fit(Xtr, ytr)
        # Get the ROC curve for the classifier
        fpr, tpr, thresholds = metrics.roc_curve(ytst, clf.predict_proba(Xtst)[:, 1])
        # Get the AUC
        roc_auc = metrics.auc(fpr, tpr)

        # Plot ROC curve
        legstr = '{:s}, AUC = {:.2f}'.format(descr, roc_auc)
        plt.plot(fpr, tpr, color, label=legstr, lw=2)

        print('## model: ', clf.__repr__(), '\n', 'AUC =', roc_auc)

    # Decorate the graph
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    _draw()


def plot_learning_curve(tr_sizes, tr_errors, tst_errors, tr_color='b', tst_color='r'):
    """Plot the learning curve from pre-computed data."""

    fig, ax = plt.subplots()

    ax.plot(tr_sizes, tr_errors, lw=2, c=tr_color, label='training error')
    ax.plot(tr_sizes, tst_errors, lw=2, c=tst_color, label='cross validation '
                                                           'error')
    ax.set_xlabel('training set size')
    ax.set_ylabel('error')

    ax.legend(loc=0)
    ax.set_xlim(0, np.max(tr_sizes))
    ax.set_ylim(0, 1)
    ax.set_title('Learning Curve for a model (fixed parameters)')
    ax.grid(True)
    _draw()


def plot_validation_curve(param_values, tr_errors, tst_errors, tr_color='b', tst_color='r'):
    """Plot the validation curve from pre-computed data."""

    fig, ax = plt.subplots()

    plt.semilogx(param_values, tr_errors, lw=2, c=tr_color, label='training error')
    plt.semilogx(param_values, tst_errors, lw=2, c=tst_color, label='cross validation '
                                                           'error')
    ax.set_xlabel('parameter value')
    ax.set_ylabel('error')

    ax.legend(loc=0)
    ax.set_xlim(0, np.max(param_values))
    ax.set_ylim(0, 1)
    ax.set_title('Validation Curve (vary model parameters)')
    ax.grid(True)
    _draw()


def compute_learning_curve(model, tr_sizes, Xtr, ytr, Xtst, ytst):
    """Compute a single learning curve."""

    # Pre-allocate the arrays for computed errors
    tr_errors = np.zeros_like(tr_sizes, dtype=np.float)
    tst_errors = np.zeros_like(tr_sizes, dtype=np.float)

    # FIXME - Task 14: Computation of a learning curve

    raise NotImplementedError
    # <YOUR CODE HERE>

    return tr_errors, tst_errors


def compute_validation_curve(model, param_name, param_range, Xtr, ytr, Xtst, ytst):
    """Compute a single learning curve."""

    tr_errors = np.zeros_like(param_range, dtype=np.float)
    tst_errors = np.zeros_like(param_range, dtype=np.float)

    # FIXME - Task 17: Computation of a validation curve
    raise NotImplementedError
    # <YOUR CODE HERE>

    return tr_errors, tst_errors


def _draw():
    """A helper method for pseudo-interactive plotting.
    """
    plt.draw()
    plt.pause(.001)
