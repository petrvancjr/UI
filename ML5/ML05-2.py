"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Decision trees and ensemble learning

Petr Posik, Jiri Spilka, CVUT, Praha 2018
"""

from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from matplotlib import pyplot as plt
import pandas as pd

import plotting


def load_data_binary_classification():
    """Load 'auto-mpg.csv' and return the disp and hp columns as numpy arrays.
    """
    df = pd.read_csv('auto-mpg.csv', skipinitialspace=True,
                     names=['mpg', 'cyl', 'disp', 'hp', 'wgt', 'acc', 'year', 'orig', 'name'])
    X = df.xs(['disp', 'hp'], axis=1).values
    y = df.orig.values
    y[y > 1] = 0
    return X, y


# Load data
X, y = load_data_binary_classification()

print('\nThe dataset (X and y) size:')
print('X: ', X.shape)
print('y: ', y.shape)
print('sum(y = 0): ', sum(y == 0))
print('sum(y = 1): ', sum(y == 1))

# random_state=2017 - lets keep the training and test set the same
Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.2, random_state=2017)

# FIXME Task 6: Model ensemble via majority voting
m1 = DecisionTreeClassifier()
m2 = AdaBoostClassifier()
m3 = RandomForestClassifier()

# raise NotImplementedError
# <YOUR CODE HERE>

plt.show()  # show whatever we have to show, it will stay open because we turned interactive mode off