# -*- coding: utf-8 -*-
"""
    ML02.py
    ~~~~~~~

    Linear regression
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from scipy import stats

import linreg


# # Load data for simple regression
X, y = linreg.load_data_for_univariate_regression()
print('\n==Simple regression==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

linreg.homogenize(X)

# # Plot the input data
linreg.plot_2d_data(X, y)


# FIXME Task 1: Estimate the model parameters by hand. <PUT YOUR ESTIMATE OF w_0 AND w_1 HERE>
w_0 = 0
w_1 = 1
wguess = np.array([w_0, w_1])
linreg.plot_predictions(wguess, c='g', label='guess')
plt.show()


# # Compute the error of the model
print('\nThe hand-crafted coefficients:')
print(wguess)
err = linreg.compute_cost_regr_lin(wguess, X, y)
print('The error:', err)

# # Find optimal weights by minimization of J
w1 = linreg.fit_regr_lin_by_minimization(X, y)
print('\nCoefficients found by minimization of J:')
print(w1)
err = linreg.compute_cost_regr_lin(w1, X, y)
print('The error:', err)
linreg.plot_predictions(w1, c='b', label='minimize')

# # Find optimal weights by normal equation
w2 = linreg.fit_regr_lin_by_normal_equation(X, y)
print('\nCoefficients found by normal equation:')
print(w2)
err = linreg.compute_cost_regr_lin(w2, X, y)
print('The error:', err)
linreg.plot_predictions(w2, c='r', label='normal eq.')

# # Use scikit-learn package to train the model
# FIXME Task 8: Linear regression using scikit
# <ADD THE CODE WHICH CREATES AN INSTANCE OF LINEAR REGRESSION,
# AND TRAINS IT ON THE TRAINING DATA>
lr = linear_model.LinearRegression().fit(linreg.homogenize(X), y)
lr.predict(wguess.reshape(1, -1))
w3 = np.array([lr.intercept_, lr.coef_[0]])
print('\nCoefficients found by sklearn.linear_model.LinearRegression:')
print(w3)
err = linreg.compute_cost_regr_lin(w3, X, y)
print('The error:', err)

# # Multivariate regression
X, _ = linreg.load_data_for_multivariate_regression()
print('\n==Multivariate regression==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

# FIXME Task 9: Multivariate regression
# Fit the regression model by any method
data = pd.read_csv("auto-mpg.csv")


#X = data["mpg"] + data["cyl"] + data["disp"] + data["weight"] + data["acc"]
X = data[["mpg","cyl","disp","weight","acc"]]
y = data['hp']
print(np.shape(y))
y = np.transpose([y])
print(np.shape(y))
y = linreg.homogenize(y)

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedHP = regr.predict([[18.0, 8, 307.0, 3504., 12.0]])
print("predicted HP",predictedHP[0,0])

wmulti = regr.coef_
print('\nCoefficients for multivariate regression model:')
print(wmulti)
# Compute the model error
err = (130.0 - predictedHP[0,0])**2
print('The error:')
print(err)

# different method - linear regression
print("Different method")
X, y = linreg.load_data_for_univariate_regression()
x = X[:,0]
print(np.shape(x))
print(np.shape(y))

slope, intercept, r, p, std_err = stats.linregress(x, y)

print("relationship x,y r: ", r)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

# printing
plt.clf()
#fig = plt.figure()
#ax1 = fig.add_axes((0.1, 0.1, 0.7, 0.7))
#ax1.set_title("Different approach")
#ax1.set_xlabel('X-axis')
#ax1.set_ylabel('Y-axis')
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()




