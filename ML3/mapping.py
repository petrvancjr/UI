""" mapping.py
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Module containing mapping functions.
"""

import numpy as np
from itertools import combinations_with_replacement

# noinspection PyPep8Naming
class PolynomialMapping:

    def __init__(self, max_deg=2):
        self.max_deg = max_deg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Map the input data x into space of polynomials.
        """
        m = np.shape(X)[0]
        D = np.shape(X)[1]
        #print("X shape", np.shape(X))

        # X = [m x D]
        #final = np.ones((m, self.max_deg*D))
        final = X
        for i in range(2, self.max_deg+1, 1):
            #final[:, i-1] = (X**i)[:,0]
            final = np.hstack((final, X**i))
        # ret [m x self.max_deg*D]
        #print("final",final)
        return final


# noinspection PyPep8Naming
class FullPolynomialMapping:

    def __init__(self, max_deg=2):
        self.max_deg = max_deg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, max_deg=2):
        """Map X to fully polynomial space, including all cross-products."""
        final = X
        c = np.shape(X)[0]
        r = np.shape(X)[1]

        for i in range(2, self.max_deg+1,1):
            combs = combinations_with_replacement(np.arange(r),i)
            for j in combs:
                Xelem = np.ones((1,c))
                for k in j:
                    Xelem = np.multiply(Xelem,np.transpose(X[:,k]))

                #print("final",np.shape(final),"Xelem",np.shape(np.transpose(Xelem)))
                final = np.hstack((final, np.transpose(Xelem)))


        return final





