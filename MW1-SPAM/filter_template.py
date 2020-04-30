"""
B(E)3M33UI - Support script for the first semestral task
"""

from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# you can do whatever you want with the these data
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
    """Return a trained spam filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2

    vec = CountVectorizer()
    clf = DummyClassifier(strategy='most_frequent')
    pipe = Pipeline(steps=[
        ('vectorizer', vec),
        ('classifier', clf)])
    pipe.fit(X, y)
    return pipe


def predict(filter1, X):
    """Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return filter1.predict(X)


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
    # X = X_train.copy()
    # X.extend(X_test)
    # y = np.hstack((y_train, y_test))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train the filter
    filter1 = train_filter(X_train, y_train)

    # Compute predictions for training data and report our accuracy
    y_tr_pred = predict(filter1, X_train)
    print('Modified accuracy on training data: ',
          modified_accuracy(y_train, y_tr_pred))

    # Compute predictions for testing data and report our accuracy
    y_tst_pred = predict(filter1, X_test)
    print('Modified accuracy on testing data: ',
          modified_accuracy(y_test, y_tst_pred))
