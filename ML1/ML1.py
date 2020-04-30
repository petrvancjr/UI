import numpy as np
from itertools import product
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)


'''
A person that comes to the general practitioner (GP) may be healthy, or may have a cold o r a flu.
GP has only one type of observation, the measurement of the body temperature divided into discrete
intervals: it can be 37, 37-38, 38-39, or over 39* degrees Celsius. GP must decide whether the
person needs no cure at all, or just needs tea and sweat, or some kind of medicine drugs, or GP can
send the person to a specialist.
'''

# Task 1 - Define the sets X, K, and D, and represent them as lists X, K, and D in Python.

X = ['below_37', '37-38', '38-39', 'over_39']
K = ['healthy', 'cold', 'flu']
D = ['no', 'tea', 'drugs', 'spec']

'''
Bayesian formulation. Task 2 - Formulate task in the Bayesian framework. What other information do we need?
To solve the task in Bayesian framework, we need to know the joint probability
distribution pxk: X x K -> <0,1>, and the cost matrix W: K x D -> R. The task is to find a strategy
q: X -> D such that the risk of the strategy R(q) is minimal.

'''

'''
Task 4 - Implement a helper function load_data(filename) to load the data from file (now for
the joint distribution, later usable for the penalty matrix).
The function shall return a dictionary, where the first 2 columns of the CSV file contain the
keys, and the third column contains the values.
'''


def load_data(filename):
    data = {}
    first_line = True  # Discard first line of csv file
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            if first_line:
                first_line = False
            else:
                # List comprehension, see DIP3
                parts = [x.strip() for x in line.split(',')]
                key = tuple(parts[:-1])
                value = float(parts[-1])
                data[key] = value
    return data


'''
Task 5: Find in the documentation what the zip() function does and how it can be used when
constructing a dictionary. Try out the following code. It may come handy in the next tasks.
Solution: If you have two lists, one containing keys and the other containing values, using the zip()
function in the dict() constructor allows you to easilly construct a dictionary from keys and
values.
'''

keys = ['k1', 'k2', 'k3']
values = ['v1', 'v2', 'v3']
print("Testing command zip:", list(zip(keys, values)))
# Out:  [('k1', 'v1'), ('k2', 'v2'), ('k3', 'v3')]

d = dict(zip(keys, values))
print("Dictionary from zip:", d)
# Out: {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}

'''
Task 6: Implement a set of functions allowing you to easilly compute the marginal and conditional distributions based on pXK.
In the following, we will have to iteratively add several values to individual dictionary items.
In such situation an error often occurs, that when adding the first value, the dictionary item does
not exist at all. Python offers several solutions to this:
• you can pre-initialize the dictionary with correct starting values (0.0 in our case),
• you can use the dict.get(key, default_value) method, which returns the value for the
key, if the key exists in the dictionary, otherwise it returns the default_value, or
• you can use the collections.DefaultDict class which allows you to specify how it should
initialize the nonexisting items
'''


def get_pX(pXK):
    pX = {}
    for x, k in pXK:
        pX[x] = pX.get(x, 0.0) + pXK[x, k]
    return pX


def get_pK(pXK):
    pK = {}
    for x, k in pXK:
        pK[k] = pK.get(k, 0.0) + pXK[x, k]
    return pK


def get_pXgK(pXK):
    pXgK = {}
    pK = get_pK(pXK)
    for x, k in pXK:
        pXgK[x, k] = pXK[x, k] / pK[k]
    return pXgK


def get_pKgX(pXK):
    pKgX = {}
    pX = get_pX(pXK)
    for x, k in pXK:
        pKgX[k, x] = pXK[x, k] / pX[x]
    return pKgX


pXK = load_data("ML01_pXK.csv")
print("pXK")
pp.pprint(pXK)

pX = get_pX(pXK)
print("pX")
pp.pprint(pX)

sumpX = sum(pX.values())
print("Sum pX", sumpX)

pK = get_pK(pXK)
print("pK")
pp.pprint(pK)

pXgK = get_pXgK(pXK)
print("pXgK")
pp.pprint(pXgK)

pKgX = get_pKgX(pXK)
print("pKgX")
pp.pprint(pKgX)

'''
Task 7: Design your own penalty matrix W : K × D → R and store it in exB_W.csv. Load the
data into variable W.
'''

W = load_data('ML01_W.csv')
print("W")
pp.pprint(W)

'''
Task 8: Make sure you understand what a strategy is in the Bayesian formulation. For us a
strategy shall be represented by a dictionary, so that we can ask q[x] (What is the decision for
observation x?). Create function make_strategy() which takes a list of possible observations and
a list of corresponding decisions, and creates a strategy, i.e. a dictionary.
'''


def make_strategy(obs, dec):
    return dict(zip(obs, dec))


'''
Task 9: How many different strategies q : X → D are there?
Solution 9: A strategy q is a function which assigns one of the possible decisions to each
possible observation. The number of strategies is thus |Q| = |D|^|X|, i.e.
'''

n_strategies = len(D)**len(X)
print("n_strategies", n_strategies)

'''
Task 10: Given all the strategies will have the same keys, we can represent them (at first) just as
a tuple of decisions. All the possible strategies can be generated using the itertools.product()
function. Generate a list of all possible 4-tuples of decisions.
'''

strategies_as_tuples = list(product(D, D, D, D))
# Just look at the first 10 strategies
print("strategies_as_tuples[:10]")
pp.pprint(strategies_as_tuples[:10])
print("len(strategies_as_tuples)", len(strategies_as_tuples))

'''
Task 11: Create function make_list_of_strategies(), which takes the list of possible observations X,
and the list of possible decisions D, and produces a list of all possible strategies, i.e. list
of dictionaries.
'''


def make_list_of_strategies(X, D):
    # This is a list comprehension, find it in Dive into Python
    decision_sets = [D for x in X]
    strategies_as_tuples = list(product(*decision_sets))
    # Another list comprehension
    return [make_strategy(X, tup) for tup in strategies_as_tuples]


strategies = make_list_of_strategies(X, D)
print("strategies[:10]")
pp.pprint(strategies[:10])  # Look at the first 10 strategies

'''
Task 12: Create function print_strategy() which takes a strategy (dictionary) q and a list of
observations (keys) X, and pretty-prints the strategy so that the order of keys is such as specified
in X.
'''


def print_strategy(q, X):
    for key in X:
        print('{:10s} {:s}'.format(key, q[key]))


print_strategy(strategies[9], X)

'''
 2.1. Bayesian strategy via complete search.
 Task 13: Create function risk() which returns the risk for a particular strategy q: R(q) =
∑x∈X ∑k∈K pXK(x, k) · W(k, q(x)) What other inputs does the function need?
'''


def risk(q, pXK, W):
    risk = 0
    for x, k in pXK:
        risk += pXK[x, k] * W[k, q[x]]
    return risk


risk_strategy9 = risk(strategies[9], pXK, W)
print("risk for strategy 9:", risk_strategy9)

pp.pprint(strategies[9])

'''
Task 14: Create function find_bayesian_strategy(). What inputs does the function need?
Go through all possible strategies, compute their risks, find the strategy with the minimal risk.
Return a 2-tuple: the Bayesian strategy and its risk.
You may find functions numpy.argmax() and numpy.argmin() useful.
'''
print("Task 14: Find Bayesian Strategy")


def find_bayesian_strategy(X, K, D, pXK, W):
    risks = []
    strategies = make_list_of_strategies(X, D)
    # Compute risks for all strategies
    for q in strategies:
        r = risk(q, pXK, W)
        risks.append(r)
    # Choose the strategy with minimal risk
    imin = np.argmin(risks)
    return strategies[imin], risks[imin]


q_bs, risk_bs = find_bayesian_strategy(X, K, D, pXK, W)
print("Computed risk", risk_bs)
print_strategy(q_bs, X)

'''
2.2. Bayesian strategy via partial risks.
We do not need to search the whole space of possible strategies. If we use partial risks, we can
construct the whole Bayesian strategy by choosing the optimal decision for each observation one
by one.
Task 15: Create function partial_risk(), which returns the partial risk for a particular decision
 d and observation x: R(d, x) = ∑k∈K pK|X(k|x) · W(k, d). What other inputs are needed?
'''
print("Task 15: Find Bayesian Strategy via Partial Risks")


def partial_risk(d, x, K, pKgX, W):
    pr = 0
    for k in K:
        pr += pKgX[k, x] * W[k, d]
    return pr


'''
Task 16: Create function find_bayesian_strategy_via_partial_risks(). For each observation,
compute the partial risk of all decisions, and assign the decision with the minimal partial
risk. Return a 2-tuple: the optimal strategy and its risk.
Solution 16:
First, let’s create a function find_optimal_decision_for_observation() which will produce
the optimal decision for an obervation and the partial risk related to it.
'''


def find_optimal_decision_for_observation(x, D, K, pKgX, W):
    prisks = []
    # Given the observation x, find the partial risk of all decisions
    for d in D:
        pr = partial_risk(d, x, K, pKgX, W)
        prisks.append(pr)
    # Find the optimal decision for the given observation
    imin = np.argmin(prisks)
    return D[imin], prisks[imin]


def find_bayesian_strategy_via_partial_risks(X, K, D, pXK, W):
    pX = get_pX(pXK)
    pKgX = get_pKgX(pXK)
    q = {}
    risk = 0
    # For each observation, find the optimal decision separately
    for x in X:
        d_opt, prisk_opt = find_optimal_decision_for_observation(
            x, D, K, pKgX, W)
        # Make the optimal decision part of the strategy
        q[x] = d_opt
        risk += pX[x] * prisk_opt
    return q, risk


q_bs2, risk2 = find_bayesian_strategy_via_partial_risks(X, K, D, pXK, W)
print("Risk via Partial Risks", risk2)
print_strategy(q_bs2, X)

'''
 3. Estimating the hidden state
Let’s move to a different task - estimating the hidden state K, i.e. D = K.
Task 17: Can the physician say anything about the hidden state of a patient before she actually
sees the patient?
Solution 17: Well, yes, using the prior probabilities of states K the physician can identify the
most probable state:
'''
print("Task 17: Estimating hidden state")

print("pK")
pp.pprint(pK)

'''
Task 18: If the physician learns a new information about the patient – the body temperature X,
she should update her beliefs and maybe change her estimate. Make sure you understand what a
strategy is in this case. How many different strategies q : X → K are there? Can you create their
list?
'''

n_strategies = len(K)**len(X)
print("n_strategies", n_strategies)

strategies = make_list_of_strategies(X, K)
pp.pprint(strategies[:5])  # Just look at the first 5 strategies

print("len strategies check", len(strategies))

'''
 3.1. MAP estimation
We are still in the field of Bayesian formulation, but with D = K, and with square matrix W
containing all ones and zeros only on its diagonal.
Task 19: Implement function find_MAP_strategy() which returns a strategy that will provide
for each observation x an estimate of the hidden state k with minimal probability of error.
Solution 19: The optimal strategy which estimates the hidden state k with a minimal probability
of error is the same as strategy that classifies the observation into the class with maximal
posterior probability, i.e. it is given by q(x) = arg maxk∈K pK|X(k|x).
'''


def find_MAP_strategy(X, K, pKgX):
    q = {}
    for x in X:
        pK_for_x = [pKgX[k, x] for k in K]
        imax = np.argmax(pK_for_x)
        q[x] = K[imax]
    return q


q = find_MAP_strategy(X, K, pKgX)
print_strategy(q, X)

'''
 3.2. Minimax formulation
Now, we leave the world of Bayesian formulation of the decision task.
We still want to estimate the object state K based on the observation X. The strategy should
assign a state to each observation with the aim to minimize the maximal probabilities of wrong
decision across all true states. We will need only the conditional probabilities pX|K; pK and W are
not required.
10
Task 20: Implement function find_minimax_strategy() which returns such a strategy that
minimizes the maximal probability of wrong decisions across all possible states.
Solution 20:
First, let’s define a function compute_max_wd_prob_for_strategy(). The result of this function
is used as a score for a strategy.
'''


def compute_max_wd_prob_for_strategy(q, X, K, pXgK):
    wd_probs = []
    for k in K:
        wd_prob_for_k = sum(pXgK[x, k] for x in X if q[x] != k)
        wd_probs.append(wd_prob_for_k)
    # Return the maximum probability of a wrong decision
    return np.max(wd_probs)


def find_minimax_strategy(X, K, pXgK):
    max_wd_probs = []
    strategies = make_list_of_strategies(X, K)
    for q in strategies:
        # Compute probability of wrong decision for all states k
        max_wd_prob = compute_max_wd_prob_for_strategy(q, X, K, pXgK)
        max_wd_probs.append(max_wd_prob)
    # Among all strategies, find the one with minimal max_wd_prob
    imin = np.argmin(max_wd_probs)
    return strategies[imin]


q_mm = find_minimax_strategy(X, K, pXgK)
print_strategy(q_mm, X)

# end
#
#
