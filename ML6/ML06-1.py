"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Neural Networks

Petr Posik, CVUT, Praha 2017
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import sys

from plotting import plot_2D_class_model, plot_xy_classified_data, plot_2D_class_model, plot_2D_class_model_from_data

# Parameter of learning
alpha = 0.01
epoch = 300

LEARNING = True

def main():
    global LEARNING
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        LEARNING = False

    # Load data
    data = np.loadtxt('data_1.csv', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].astype(int)

    print('\nThe dataset (X and y) size:')
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('sum(y = 0): ', sum(y == 0))
    print('sum(y = 1): ', sum(y == 1))

    print("Task 1 - Backpropagation")
    if LEARNING:
        print("Learning is on. To run with learned values, run with parameter 'plot'")
        w = my_learning(X,y)
    else: # use already learned values
        w = {'w10': -9.82194232366161, 'w11': 14.341738648673063, 'w12': 11.062726936361718, 'w20': -1.4287690817030179, 'w21': -14.519841028793476, 'w22': 11.265280284383621, 'w30': -8.833017167589208, 'w31': 18.67426152449217, 'w32': -19.462692421361748}

    y_ = forward(X, w)
    plot_xy_classified_data(X[:, 0], X[:, 1], np.transpose(vector_round(y_)))
    plot_2D_class_model_from_data(forward, w)

    print("Press Enter")
    input()
    plt.close()

    ### -------------------------------- ###
    print("Task 2 - MLP Classifier")
    # MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-2, activation='logistic',
                     hidden_layer_sizes=(2,), random_state=1)
    clf.fit(X, y)
    y_ = clf.predict(X)
    print("MLP finished")
    # Plot with decision boundary
    plot_xy_classified_data(X[:, 0], X[:, 1], np.transpose(vector_round(y_)))
    plot_2D_class_model(clf)

    print("press Enter")
    input()
    plt.close()
    ### --------------------------------- ###
    print("MLP Classifier - XOR dataset")

    data = np.loadtxt('data_xor_rect.csv', delimiter=',')
    print(data.shape)

    X = data[:, 0:2]
    y = data[:, 2].astype(int)

    print('\nThe dataset (X and y) size:')
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('sum(y = 0): ', sum(y == 0))
    print('sum(y = 1): ', sum(y == 1))

    # XOR dataset - MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-1, activation='relu', max_iter=10000,
                     hidden_layer_sizes=(100,40,))
    clf.fit(X, y)
    y_ = clf.predict(X)
    print("MLP finished")
    # Plot with decision boundary

    print("zoom out")
    plot_xy_classified_data(X[:, 0], X[:, 1], np.transpose(vector_round(y_)), margin=True)
    plot_2D_class_model(clf)

    print("Enter to exit")
    input()


def my_learning(X,y):
    print("epoch: ", epoch)
    print("learning rate: ", alpha)

    # Task 6: Implement backpropagation
    w = {
        'w10': np.random.rand(),
        'w11': np.random.rand(),
        'w12': np.random.rand(),
        'w20': np.random.rand(),
        'w21': np.random.rand(),
        'w22': np.random.rand(),
        'w30': np.random.rand(),
        'w31': np.random.rand(),
        'w32': np.random.rand()
    }

    # Epoch loop
    for i in range(epoch):
        # through whole data set, update weights everytime
        for j in range(0, np.shape(X)[0]):
            # update weights
            w = backpropagate(w, X[j], y[j])

        # Summary every epochs
        y_ = forward(X, w)
        print("epoch", i, "loss_E", loss_E(X, w, y))

    return w


def sigmoid(a):
    return 1/(1+np.exp(-a))


def derivace_sigmoid(a):
    return sigmoid(a) * (1-sigmoid(a))


def my_superFunc_calculate_TRUERATE(X, y, y_):
    truerate = 0
    for i in range(0, np.shape(X)[0]):
        if y[i] == y_[i]:
            truerate += 1

    plot_xy_classified_data(X[:, 0], X[:, 1], vector_round(y_))

    return truerate


def vector_round(y):
    y_ = []
    for i in y:
        y_.append(int(round(i)))
    return y_


def backpropagate(w, x, y):
    ''' output: weights w; size(w) = 9
    w <- w - alpha*grad(E(w))

    '''
    w_new = deepcopy(w)

    z3, z2, z1 = forward_single(x, w)
    # y_hat = z3
    x1 = x[0]
    x2 = x[1]

    dE_dw30 = -(y - z3)*1
    dE_dw31 = -(y - z3)*z1
    dE_dw32 = -(y - z3)*z2
    if dE_dw31 != np.nan and dE_dw32 != np.nan and dE_dw30 != np.nan:
        w_new['w31'] = w['w31'] - alpha * dE_dw31
        w_new['w32'] = w['w32'] - alpha * dE_dw32
        w_new['w30'] = w['w30'] - alpha * dE_dw30

    dd = derivace_sigmoid(w['w10']*1+w['w11']*x1+w['w12']*x2)
    dE_dw10 = -(y - z3)*w['w31']*dd*1
    dE_dw11 = -(y - z3)*w['w31']*dd*x1
    dE_dw12 = -(y - z3)*w['w31']*dd*x2
    if dE_dw11 != np.nan and dE_dw12 != np.nan and dE_dw10 != np.nan:
        w_new['w11'] = w['w11'] - alpha * dE_dw11
        w_new['w12'] = w['w12'] - alpha * dE_dw12
        w_new['w10'] = w['w10'] - alpha * dE_dw10

    dd = derivace_sigmoid(w['w20']*1+w['w21']*x1+w['w22']*x2)
    dE_dw20 = -(y - z3)*w['w32']*dd*1
    dE_dw21 = -(y - z3)*w['w32']*dd*x1
    dE_dw22 = -(y - z3)*w['w32']*dd*x2
    if dE_dw21 != np.nan and dE_dw22 != np.nan and dE_dw20 != np.nan:
        w_new['w21'] = w['w21'] - alpha * dE_dw21
        w_new['w22'] = w['w22'] - alpha * dE_dw22
        w_new['w20'] = w['w20'] - alpha * dE_dw20

    return w_new


def loss_E(X, w, y):
    '''Cross entropy loss function
    With respect to weights

    '''
    y_hat = forward(X, w)
    E = 0.0
    for i in range(0, np.shape(X)[0]):
        tmp = np.power(y[i] * np.log10(y_hat[i])
                       - (1-y[i]) * np.log10(1-y_hat[i]), 2)
        if tmp != np.nan:
            E += tmp

    return E


def forward_single(x, w):

    z1 = sigmoid(w['w10'] + w['w11'] * x[0] + w['w12'] * x[1])
    z2 = sigmoid(w['w20'] + w['w21'] * x[0] + w['w22'] * x[1])

    z3 = sigmoid(w['w30'] + z1 * w['w31'] + z2 * w['w32'])

    return z3, z2, z1


def forward(X, w):
    y = []
    for n, x in enumerate(X):
        z1 = sigmoid(w['w10'] + w['w11'] * x[0] + w['w12'] * x[1])
        z2 = sigmoid(w['w20'] + w['w21'] * x[0] + w['w22'] * x[1])

        z3 = sigmoid(w['w30'] + z1 * w['w31'] + z2 * w['w32'])
        y.append(z3)

    return y


# catch all exceptions
try:
    plt.ion()  # turn interactive mode on so our plots stay open like in matlab
    main()


except BaseException as e:
    import traceback
    traceback.print_exc()  # mimic printing traceback from an exception
    plt.ioff()  # turn interactive mode off
    print("base excepttion")
    plt.show()  # show whatever we have to show, it will stay open because we
    # turned interactive mode off


'''
Mistakes I made:
    rounding only when plotting or getting final result
    error in backpropagation

'''
