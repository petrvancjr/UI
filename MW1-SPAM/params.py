import numpy as np
import itertools
import sys
'''
Comprehensive dictionary about global parameters for sklearn library methods
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
    'max_df': np.linspace(0, 1, 11),
    'min_df': np.linspace(0, 1, 11)
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
    'max_fun': [15000],
    'max_df': np.linspace(0, 1, 11),
    'min_df': np.linspace(0, 1, 11)
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
    'power_t': [0.5],
    'max_df': np.linspace(0, 1, 11),
    'min_df': np.linspace(0, 1, 11)
    #early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False
},{ # best - default-bayes
    'ngram_range': [(1, 1)],
    'max_df': [0.4],
    'min_df': [0]
},{ # best - bayes - 0.856
    'alpha': [0.1],
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
    'max_df': [1.0],
    'min_df': [1]
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
    'max_df': [1.0],
    'min_df': [1]
    #early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False
}]
