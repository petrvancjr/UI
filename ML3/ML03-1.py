# -*- coding: utf-8 -*-
"""
    ML03-1.py
    ~~~~~~~~~

    Basis expansion with linear regression
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.pipeline import Pipeline


import mapping
import ml03_utils

# Load data for simple regression
X, y = ml03_utils.load_data_for_univariate_regression()
print('\n== Univariate regression ==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

# Plot the input data
ml03_utils.plot_xy_data(X, y, xlbl='disp', ylbl='hp')

# Fit linear regression model and plot it
lm = linear_model.LinearRegression()
lm.fit(X, y)
ml03_utils.plot_1d_regr_model(lm)
plt.show()

# Compute its error
err = ml03_utils.compute_model_error(lm, X, y, ml03_utils.compute_err_mse, pos=False)
print(f'Degree 1 model: MSE = {err:.3f}')


# Fit a quadratic model and plot it
pm = mapping.PolynomialMapping()
transformed = pm.transform(X)
#print("transformed",transformed)

# some Pipe with PolynomialMapping and LinearRegressionModel
lm_ = linear_model.LinearRegression()
pipe = Pipeline([('pvm', pm), ('lrm', lm_)])
# fit the pipe to the training data
model = pipe.fit(X, y)

#print("predictions",model)
ml03_utils.plot_xy_data(X, y, xlbl='disp', ylbl='hp')
ml03_utils.plot_1d_regr_model(model)
plt.show()

# # Higher degree polynomials
ml03_utils.plot_xy_data(X, y, xlbl='disp', ylbl='hp')

# Degrees of polynomials with the colors of the lines
degrees = [(1, 'b'), (2, 'r'), (3, 'g'), (4, 'y')]
# Array for legend descriptions
legstr = []
for deg, color in degrees:
    # create instance Polynomial Mapping with respective degree
    pm = mapping.PolynomialMapping(max_deg=deg)
    #transformed = pm.transform(X)
    # create linear regression model
    lm_ = linear_model.LinearRegression()
    # fit the model
    pipe = Pipeline([('pvm', pm), ('lrm', lm_)])
    model = pipe.fit(X, y)
    # plot the model in the graph
    ml03_utils.plot_1d_regr_model(model, c=color)
    # compute error
    err = ml03_utils.compute_model_error(lm_, X, y, ml03_utils.compute_err_mse, pos=False)
    print("error", err)

plt.legend(legstr, loc='upper left')

plt.show()
