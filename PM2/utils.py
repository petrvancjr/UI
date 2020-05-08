"""
Helper functions for use in HMM projects

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""



import itertools
import bisect
import random
from collections import Counter


def normalized(P, factor=None):
    """Return a normalized copy of the distribution given as a Counter"""
    factor = factor or 1.0/sum(P.values())
    norm = Counter({k: v * factor for k, v in P.items()})
    return norm


def weighted_random_choice(pdist):
    """Choose an element from distribution given as dict of values and their probs."""
    weighted_choices = [(k,v) for k,v in pdist.items()]
    choices, weights = zip(*weighted_choices)
    cumdist = list(itertools.accumulate(weights))
    x = random.random() * cumdist[-1]
    return choices[bisect.bisect(cumdist, x)]


def test_weighted_random_choices():
    from collections import Counter
    pdist = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    c = Counter()
    for i in range(100000):
        char = weighted_random_choice(pdist)
        c[char] += 1
    print(c)


if __name__ == "__main__":
    test_weighted_random_choices()


