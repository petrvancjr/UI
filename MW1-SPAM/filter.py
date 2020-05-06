"""
B(E)3M33UI - Script for the first semestral task
"""

from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
import numpy as np
import itertools

import string
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log, sqrt
import pandas as pd
#matplotlib inline
import time
from params import *

# Needed for first run of program, so I will leave it here
nltk.download('stopwords')

# Dataset directories
TR_DATA = 'spam-data/spam-data-1'
TST_DATA = 'spam-data/spam-data-2'


def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError(
            'The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


our_scorer = make_scorer(modified_accuracy, greater_is_better=True)


def train_filter(X, y):
    """
    Naive Bayes train filter, best solution, with given parameters.
    Reaching 0.9 accuracy, depending on dataset decomposition.
    Return a trained spam filter.
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2

    # Naive Bayes Classifier
    vec = CountVectorizer(analyzer=process_data)
    clf = MultinomialNB()
    pipe = Pipeline(steps=[
        ('vectorizer', vec),
        ('classifier', clf)
        ])
    pipe.set_params(
    vectorizer__ngram_range=(1, 1),
    vectorizer__max_df= 0.4,
    vectorizer__min_df= 0,
    classifier__alpha=0.1
    )
    pipe.fit(X, y)
    return pipe


def my_train_filter(X, y, param):
    """
    My modified train filter, with parameters for training.
    param - Dictionary of training attempt
    Return a trained spam filter.
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 3

    """
    Building the pipeline with chosen method and setting the parameters
    """
    if (METHOD == 'default-bayes'):
        vec = CountVectorizer(analyzer=process_data)
        clf = DummyClassifier(strategy='most_frequent')
        pipe = Pipeline(steps=[
        ('vectorizer', vec),
        ('classifier', clf)
        ])
    elif (METHOD == 'bayes'):
        vec = CountVectorizer(analyzer=process_data)
        tfidf = TfidfTransformer()
        clf = MultinomialNB()
        pipe = Pipeline(steps=[
        ('vectorizer', vec),
        #('tfidf', tfidf),
        ('classifier', clf)
        ])
        pipe.set_params(
        vectorizer__ngram_range=param['ngram_range'],
        vectorizer__max_df=param['max_df'],
        vectorizer__min_df=param['min_df'],
        classifier__alpha=param['alpha']
        )
    elif (METHOD == 'MLP'):
        vec = CountVectorizer(analyzer=process_data)
        tfidf = TfidfTransformer()
        mlp = MLPClassifier()
        pipe = Pipeline(steps=[
        ('vectorizer', vec),
        #('tfidf', tfidf),
        ('classifier', mlp)
        ])
        pipe.set_params(
        vectorizer__ngram_range=param['ngram_range'],
        classifier__learning_rate=param['learning_rate'],
        classifier__solver=param['solver'],
        classifier__alpha=param['alpha'],
        classifier__activation=param['activation'],
        classifier__hidden_layer_sizes=param['hidden_layer_sizes'],
        vectorizer__max_df= ['max_df'],
        vectorizer__min_df= ['min_df']
        )
    elif (METHOD == 'SGD'):
        vec = CountVectorizer(analyzer=process_data)
        tfidf = TfidfTransformer()
        sgd = SGDClassifier()
        pipe = Pipeline(steps=[
        ('vectorizer', vec),
        #('tfidf', tfidf),
        ('classifier', sgd)
        ])
        pipe.set_params(
        vectorizer__ngram_range=param['ngram_range'],
        classifier__loss=param['loss'],
        classifier__penalty=param['penalty'],
        classifier__alpha=param['alpha'],
        classifier__eta0=param['eta0'],
        classifier__l1_ratio=param['l1_ratio'],
        classifier__fit_intercept= param['fit_intercept'],
        classifier__tol=param['tol'],
        classifier__random_state=param['random_state'],
        classifier__learning_rate=param['learning_rate'],
        vectorizer__max_df=param['max_df'],
        vectorizer__min_df=param['min_df']
        )
    try:
        pipe.fit(X, y)
    except ValueError:
        return None
    #print(len(vec.vocabulary_))
    return pipe


def predict(filter1, X):
    """Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return filter1.predict(X)

def process_data(message):
    # Lowercase all words
    message = message.lower()
    # Split sentences into words array
    words = word_tokenize(message)
    # Don't include short words
    words = [w for w in words if len(w) > 2]
    # Stop words
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]
    # PorterStemmer
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words

def checkDepedences(param):
    # Depedences for bayes
    if METHOD == 'bayes':
        if param['min_df']>param['max_df']:
            return True
    return False

if __name__ == '__main__':
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

    start = time.time()
    # Demonstration how the filter will be used but you can do whatever you want with the these data
    # Load training data
    data_tr = load_files(TR_DATA, encoding='utf-8')
    X_train = data_tr.data
    y_train = data_tr.target

    # Load testing data
    data_tst = load_files(TST_DATA, encoding='utf-8')
    X_test = data_tst.data
    y_test = data_tst.target

    # or you can make a custom train/test split (or CV)
    #X = X_train.copy()
    #X.extend(X_test)
    #y = np.hstack((y_train, y_test))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # info
    #numMails = np.shape(X_train)[0]
    #print("numMails", numMails)
    #X_train_len = []
    #for mail in X_train:
    #    X_train_len.append(len(mail))
    #plt.hist(X_train_len, bins=50)
    #plt.show()
    #input()
    #

    accuracyBefore = 0.0
    isBetter = True
    bestParameters = None
    bestAccuracy = 0.0
    if OPTION == 'search': # Choose all params
        Dict = Dict[METHODS.index(METHOD)]
    else: # Choose optimal params
        Dict = Dict[METHODS.index(METHOD)+len(METHODS)]
    keys, values = zip(*Dict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in params:

        if checkDepedences(param):
            continue

        print("param", param)
        filter1 = my_train_filter(X_train, y_train, param)
        if filter1 == None:
            continue
        # Compute predictions for training data and report our accuracy
        y_tr_pred = predict(filter1, X_train)
        print('Modified accuracy on training data: ',
              modified_accuracy(y_train, y_tr_pred))

        # Compute predictions for testing data and report our accuracy
        y_tst_pred = predict(filter1, X_test)
        accuracyNow = modified_accuracy(y_test, y_tst_pred)
        print('Modified accuracy on testing data: ',
              modified_accuracy(y_test, y_tst_pred))

        if (accuracyNow > bestAccuracy):
            bestParameters = param
            bestAccuracy = accuracyNow
        else:
            isBetter = False

        #print(classification_report(y_tr_pred, y_train))
        #print(classification_report(y_tst_pred, y_test))
        accuracyBefore = accuracyNow


    end = time.time()
    print("Time elapsed", end - start)
    print("best Parameters are: ", bestParameters, " with accuracy: ", bestAccuracy)
