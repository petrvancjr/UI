"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Neural Networks

Petr Posik, CVUT, Praha 2017
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

from plotting import plot_2D_class_model, plot_xy_classified_data

# Parameter of learning
alpha = 0.00001


def main():

    # Load data
    data = np.loadtxt('data_1.csv', delimiter=',')
    # data = np.loadtxt('data_xor_rect.csv', delimiter=',')

    print(data.shape)

    X = data[:, 0:2]
    y = data[:, 2].astype(int)

    print('\nThe dataset (X and y) size:')
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('sum(y = 0): ', sum(y == 0))
    print('sum(y = 1): ', sum(y == 1))

    # plot_xy_classified_data(X[:, 0], X[:, 1], y)

    # # FIXME Task 6: Implement backpropagation
    w = np.ones(9)
    y_ = forward(X, w)

    print("truerate", my_superFunc_calculate_TRUERATE(X, y, y_))

    epoch = 100
    # Epoch loop
    for i in range(epoch):
        # through whole data set, update weights everytime
        for j in range(0, np.shape(X)[0]):
            # update weights
            w = backpropagate(w, X[j], y[j])

        # Summary every epochs
        y_ = forward(X, w)
        print("epoch", i, "loss_E", loss_E(X, w, y),
              my_superFunc_calculate_TRUERATE(X, y, y_), "weights", w)

    y_ = forward(X, w)
    plot_xy_classified_data(X[:, 0], X[:, 1], np.transpose(y_))

    input()

    # # FIXME Task 7: Visualize decision boundary
    # raise NotImplementedError
    # <YOUR CODE HERE>

    # # FIXME Task 8: MLPClassifier
    # raise NotImplementedError
    # <YOUR CODE HERE>
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                 hidden_layer_sizes=(5, 2), random_state=1)

    # # FIXME Task 9: MLPClassifier
    # raise NotImplementedError
    # <YOUR CODE HERE>


def sigmoid(a):
    return 1/(1+np.exp(-a))


def my_superFunc_calculate_TRUERATE(X, y, y_):
    truerate = 0
    for i in range(0, np.shape(X)[0]):
        if y[i] == y_[i]:
            truerate += 1

    plot_xy_classified_data(X[:, 0], X[:, 1], y_)

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
    w_new = np.zeros(9)

    z3, z2, z1 = forward_single(x, w)
    # y_hat = z3
    x1 = x[0]
    x2 = x[1]

    dE_dw31 = -(y - z3)*z1
    w_new[7] = w[7] - alpha * dE_dw31
    dE_dw32 = -(y - z3)*z2
    w_new[8] = w[8] - alpha * dE_dw32
    dE_dw30 = -(y - z3)
    w_new[6] = w[6] - alpha * dE_dw30
    # dE_dw30 - error last layer -> I shall use it
    delta_var = -dE_dw30
    z1 = 0
    z2 = 0
    dE_dw11 = -(delta_var - z1)*x1
    w_new[1] = w[1] - alpha * dE_dw11
    dE_dw12 = -(delta_var - z1)*x2
    w_new[2] = w[2] - alpha * dE_dw12
    dE_dw10 = -(delta_var - z1)
    w_new[0] = w[0] - alpha * dE_dw10

    dE_dw21 = -(delta_var - z2)*x1
    w_new[4] = w[4] - alpha * dE_dw21
    dE_dw22 = -(delta_var - z2)*x2
    w_new[5] = w[5] - alpha * dE_dw22
    dE_dw20 = -(delta_var - z2)
    w_new[3] = w[3] - alpha * dE_dw20

    return w_new


def loss_E(X, w, y):
    '''Cross entropy loss function
    With respect to weights

    '''
    y_hat = forward(X, w)
    E = 0.0
    for i in range(0, np.shape(X)[0]):
        tmp = y[i] * np.log(y_hat[i]) - (1-y[i]) * np.log(1-y_hat[i])
        if tmp != np.nan:
            E += tmp
    return E


def forward_single(x, w):

    z1 = sigmoid(w[0] + w[1] * x[0] + w[2] * x[1])
    z2 = sigmoid(w[3] + w[4] * x[0] + w[5] * x[1])

    z3 = sigmoid(w[6] + z1 * w[7] + z2 * w[8])

    return np.round(z3), np.round(z2), np.round(z1)


def forward(X, w):
    y = []
    for n, x in enumerate(X):
        z1 = sigmoid(w[0] + w[1] * x[0] + w[2] * x[1])
        z2 = sigmoid(w[3] + w[4] * x[0] + w[5] * x[1])

        z3 = sigmoid(w[6] + z1 * w[7] + z2 * w[8])
        y.append(z3)

    return vector_round(y)


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
