""" ML03-2.py
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Basis expansion for linear classification. SVM.
"""

from matplotlib import pyplot as plt
from sklearn import linear_model, svm
from sklearn.pipeline import Pipeline

import mapping
import ml03_utils

# Load 2D data for classification
X, y = ml03_utils.load_data_2d_classification()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='disp', ylbl='hp')

print('\n=== Comparison of several classification models ===\n')

# # Logistic regression model with original features only
# create model
lm_ = linear_model.LogisticRegression()
# fit it to the data
logr = lm_.fit(X,y)
ml03_utils.plot_2d_class_model(logr)
plt.show()
plt.clf()
# # The error of the model
err = ml03_utils.compute_model_error(logr, X, y, ml03_utils.compute_err_01)
# Build the message and display it
msg = f'Pure logistic regression: error = {err:.3f}'
plt.title(msg)
print(msg)


# Logistic regression with purely polynomial terms
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='disp', ylbl='hp')
degree = 2

# pipeline
lm = linear_model.LogisticRegression()
pm = mapping.PolynomialMapping()
pipe = Pipeline([('pvm', pm), ('lrm', lm)])
model = pipe.fit(X,y)
ml03_utils.plot_2d_class_model(model)
err = ml03_utils.compute_model_error(model[-1], X, y, ml03_utils.compute_err_01)

# Build the message and display it
msg = f'Logistic regression with pure polynomials (deg = {degree}): error = {err:.3f}'
plt.title(msg)
print(msg)

plt.show()
plt.clf()

# Logistic regression with fully polynomial mapping
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='disp', ylbl='hp')
degree = 2

# pipeline
lm = linear_model.LogisticRegression()
pm = mapping.FullPolynomialMapping()
pipe = Pipeline([('pvm', pm), ('lrm', lm)])
model = pipe.fit(X,y)
ml03_utils.plot_2d_class_model(model)
err = ml03_utils.compute_model_error(model[-1], X, y, ml03_utils.compute_err_01)

# Build the message and display it
msg = f'Logistic regression with full polynomial mapping (deg = {degree}): error = {err:.3f}'
plt.title(msg)
print(msg)

plt.show()
plt.clf()

# Support vector classification with linear kernel
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='disp', ylbl='hp')

# instantiate sklearn.svm.SVC with linear kernel
svmm = svm.SVC(kernel='linear',gamma='auto')
# fit it to the data
model = svmm.fit(X,y)
# plot the classification of the model
ml03_utils.plot_2d_class_model(model)
# compute error
err = ml03_utils.compute_model_error(model, X, y, ml03_utils.compute_err_01)

# Build the message and display it
msg = f'SVM with linear kernel: error = {err:.3f}'
plt.title(msg)
print(msg)

plt.show()
plt.clf()

# Support vector classification with RBF kernel
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='disp', ylbl='hp')

# instantiate sklearn.svm.SVC with linear kernel
svmm = svm.SVC(gamma=0.0001)
# fit it to the data
model = svmm.fit(X,y)
# plot the classification of the model
ml03_utils.plot_2d_class_model(model)
# compute error
#err = ml03_utils.compute_model_error(model, X, y, ml03_utils.compute_err_01)
# Build the message and display it
msg = f'SVM with RBF kernel: error = {err:.3f}'
plt.title(msg)
print(msg)

plt.show()
