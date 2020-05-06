import numpy as np
import itertools
import sys
'''
Comprehensive dictionary about global parameters for sklearn library
'''
METHODS = ['default-bayes', 'bayes', 'MLP', 'SGD']
OPTIONS = ['best', 'search']
Dict = [
{ # default-bayes
    'ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
    'max_df': np.linspace(0, 1, 11),
    'min_df': np.linspace(0, 1, 11)
},{ # bayes
    'alpha': [0.1,0.01,0.001],
    'ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
    'max_df': [0.4], #np.linspace(0, 1, 11),
    'min_df': [0] #np.linspace(0, 1, 11)
},{ # MLP
    'ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
    'hidden_layer_sizes': [(100, )],
    'activation': ['tanh'], #['identity','logistic','tanh','relu'],
    'solver': ['adam'], #['lbfgs','sgd','adam'],
    'alpha': [0.1,0.01,0.001,0.0001,0.00001],
    'batch_size': ['auto'],
    'learning_rate': ['adaptive'], #['constant','invscaling','adaptive'],
    'learning_rate_init': [0.001], #[0.01,0.001, 0.0001],
    'power_t': [0.5],
    'max_iter': [200],
    'shuffle': [True],
    'random_state': [None],
    'tol': [0.0001],
    'verbose': [False],
    'warm_start': [False],
    'momentum': [0.9],
    'nesterovs_momentum': [True],
    'early_stopping': [False],
    'validation_fraction': [0.1],
    'beta_1': [0.9],
    'beta_2': [0.999],
    'epsilon': [1e-08],
    'n_iter_no_change': [10],
    'max_fun': [15000]
},{ # SGD
    'ngram_range': [(2,2)],
    'loss': ['hinge', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l1','l2', 'elasticnet'],
    'alpha': [0.1,0.01,0.001,0.0001],
    'l1_ratio': [0.15],
    'fit_intercept': [True],
    'max_iter': [1000],
    'tol': [0.001],
    'shuffle': [True],
    'verbose': [0],
    'epsilon': [0.1],
    'n_jobs': [None],
    'random_state': [None],
    'learning_rate': ['optimal'],
    'eta0': [0.0,0.1,0.2],
    'power_t': [0.5]
    #early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False
},{ # best - default-bayes
    'ngram_range': [(1, 1)],
    'max_df': [0.4],
    'min_df': [0]
},{ # best - bayes - 0.856
    'alpha': [0.01],
    'ngram_range': [(1, 1)],
    'max_df': [0.4],
    'min_df': [0]
},{ # best - MLP - 0.837
    'ngram_range': [(1, 2)],
    'hidden_layer_sizes': [(100, )],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.001],
    'batch_size': ['auto'],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.001],
    'power_t': [0.5],
    'max_iter': [200],
    'shuffle': [True],
    'random_state': [None],
    'tol': [0.0001],
    'verbose': [False],
    'warm_start': [False],
    'momentum': [0.9],
    'nesterovs_momentum': [True],
    'early_stopping': [False],
    'validation_fraction': [0.1],
    'beta_1': [0.9],
    'beta_2': [0.999],
    'epsilon': [1e-08],
    'n_iter_no_change': [10],
    'max_fun': [15000],
    'max_df': [0.4],
    'min_df': [0]
},{ # best - SGD - 0.86
    'ngram_range': [(2,2)],
    'loss': ['modified_huber'],
    'penalty': ['elasticnet'],
    'alpha': [0.001],
    'l1_ratio': [0.15],
    'fit_intercept': [True],
    'max_iter': [1000],
    'tol': [0.001],
    'shuffle': [True],
    'verbose': [0],
    'epsilon': [0.1],
    'n_jobs': [None],
    'random_state': [None],
    'learning_rate': ['optimal'],
    'eta0': [0.1],
    'power_t': [0.5],
    'max_df': [0.4],
    'min_df': [0]
    #early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False
}]

# globals
state = 0
currentN = 0

# Move Based on isBetter boolean
# param is array with values of parameter


def moveInParams(param, isBetter):
    global state
    global currentN
    quitting = False
    # Need to remember... for given variable if go right, left or continue -> max
    #                               state           1       2       3
    if state == 0:  # init
        state += 1
        # Return mean value
        currentN = int(len(param)/2)+1
        return currentN, param[currentN], quitting

    # -1 - left, 1 - right, 0 - no movement
    movement = 0

    if isBetter:
        # Return next value
        if state == 1:
            movement = 1
        if state == 2:
            movement = -1
    else:
        state += 1
        # Go left or quit + revert changes
        if state == 2:
            movement = -2  # Revert changes and go left -> -2
        if state == 3:
            movement = -1  # Only revert changes -> -1

    n_p_val, n_param_val, state = move(param, movement, state)

    if state == 3:
        quitting = True
    return n_p_val, n_param_val, quitting


def move(param, movement, state):
    global currentN
    if (currentN+movement) < 0 or (currentN+movement) >= len(param):
        state = 3
        return currentN, param[currentN], state
    currentN = currentN+movement
    return currentN, param[currentN], state


if __name__ == "__main__":
    '''
    Choose the method
    '''
    if len(sys.argv)>=2 and sys.argv[1] in METHODS:
        METHOD = sys.argv[1]
    else:
        METHOD = 'MLP'
    print("Method: ", METHOD)
    '''
    Option: 'best' - get best solution, 'search' - search whole space
    '''
    if len(sys.argv)>=3 and sys.argv[2] == "search":
        OPTION = 'search'
    else:
        OPTION = 'best'

    print("Option: ", OPTION)
    if OPTION == 'search': # Choose all params
        Dict = Dict[METHODS.index(METHOD)]
    else: # Choose optimal params
        Dict = Dict[METHODS.index(METHOD)+len(METHODS)]

    keys, values = zip(*Dict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_param_assesment = []
    for opt in values:
        best_param_assesment.append(opt[int(len(opt)/2)])

    isBetter = True
    accuracyOpt = 0.0
    for n_p, param in enumerate(params):

        accuracyBefore = 0.0
        state = 0
        while True:
            n_p_val, n_param_val, quitting = moveInParams(param, isBetter)
            if quitting:
                break
            # Searching for maximum in one int variable
            tmp_prm = best_param_assesment
            # Replace tmp_prm for current value, that I'm improving
            #       ?

            tmp_prm[n_p] = round(n_param_val, 2)
            # Run training
            #train(isBetter)
            # Check improvement, predict
            # acc = predict()
            acc = tmp_prm[0]*(1-tmp_prm[1])
            if acc > accuracyBefore:
                isBetter = True

                if acc > accuracyOpt:
                    print("New best solution: ", tmp_prm,
                          "with acc: ", round(acc, 2))
                    accuracyOpt = acc
                    best_param_assesment = tmp_prm

            else:
                isBetter = False

            # Print results
            print("applied param: ", tmp_prm, " acc ", round(acc, 2))
            accuracyBefore = acc


##
#
#
#
#
#
#
#
#
#
#
