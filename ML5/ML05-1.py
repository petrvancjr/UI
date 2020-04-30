"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Decision trees and ensemble models

Petr Posik, Jiri Spilka, CVUT, Praha 2018
"""

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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

'''
Task 1: Decision Tree
'''
m1 = DecisionTreeClassifier()
m1_name = 'DT'
m1_param_grid = {'max_depth': range(1, 21)}
p1 = task_GS_val_curve(m1, m1_name, m1_param_grid, Xtr, ytr, Xtst, ytst)

# < USE TRAINING DATA ONLY >

# FIXME Task 2: Adaboost
# raise NotImplementedError
# <YOUR CODE HERE>
m2 = ...

# FIXME Task 3: Random Forest
# raise NotImplementedError
# <YOUR CODE HERE>
m3 = ...

# FIXME Task 4-5: Classification performance on test set, decision boundary
# raise NotImplementedError
# <YOUR CODE HERE>

# < USE TEST DATA FOR PREDICTIONS>

# prediction accuracy on test set
print('Accuracy on test data:')
print('Decision Tree: ', metrics.accuracy_score(ytst, m1.predict(Xtst)))
print('Adaboost: ', metrics.accuracy_score(ytst, m2.predict(Xtst)))
print('Random forest: ', metrics.accuracy_score(ytst, m3.predict(Xtst)))

plt.show()  # show whatever we have to show, it will stay open because we turned interactive mode off
