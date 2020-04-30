"""
B(E)3M33UI - Support script for the first semestral task
"""

from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log, sqrt
import pandas as pd
#matplotlib inline

nltk.download('stopwords')

# you can do whatever you want with the these data
TR_DATA = 'spam-data/spam-data-1'
TST_DATA = 'spam-data/spam-data-2'

'''
1 - Default template dummy filter
2 - Advanced naive bayes filter
'''
TYPE_FILTER = 2


def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError(
            'The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


our_scorer = make_scorer(modified_accuracy, greater_is_better=True)


def train_filter(X, y):
    """ Default train_filter from given template
    Return a trained spam filter.
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2

    # Naive Bayes Classifier
    vec = CountVectorizer()
    clf = DummyClassifier(strategy='most_frequent')
    pipe = Pipeline(steps=[
        ('vectorizer', vec),
        ('classifier', clf)])
    pipe.fit(X, y)
    return pipe


def adv_train_filter(X, y):
    """
    Advanced Naives Bayes Classifier
    Return a trained spam filter.
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2

    # Naive Bayes Classifier
    vec = CountVectorizer(analyzer=process_data_copied)
    tfidf = TfidfTransformer()
    clf = MultinomialNB()
    pipe = Pipeline(steps=[
        ('vectorizer', vec),
        ('tfidf', tfidf),
        ('classifier', clf)])
    pipe.fit(X, y)
    print(len(vec.vocabulary_))
    return pipe


def predict(filter1, X):
    """Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return filter1.predict(X)


def process_data_copied(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


def process_data(message):
    # lowercase all words
    message = message.lower()
    # Split senteces into words array
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


if __name__ == '__main__':

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
    X = X_train.copy()
    X.extend(X_test)
    y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # info
    numMails = np.shape(X_train)[0]
    print("numMails", numMails)
    X_train_len = []
    for mail in X_train:
        X_train_len.append(len(mail))
    #plt.hist(X_train_len, bins=50)
    #plt.show()
    #input()
    #

    # Train the filter
    if (TYPE_FILTER == 1):
        filter1 = train_filter(X_train, y_train)
    elif (TYPE_FILTER == 2):
        filter1 = adv_train_filter(X_train, y_train)
    else:
        # throw some exception
        print("ERROR")

    # Compute predictions for training data and report our accuracy
    y_tr_pred = predict(filter1, X_train)
    print('Modified accuracy on training data: ',
          modified_accuracy(y_train, y_tr_pred))

    # Compute predictions for testing data and report our accuracy
    y_tst_pred = predict(filter1, X_test)
    print('Modified accuracy on testing data: ',
          modified_accuracy(y_test, y_tst_pred))

    print(classification_report(y_tr_pred, y_train))
    print(classification_report(y_tst_pred, y_test))
