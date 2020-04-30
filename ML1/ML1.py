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

# Task 4 - Implement a helper function load_data(filename) to load the data from file (now for
# the joint distribution, later usable for the penalty matrix).
# The function shall return a dictionary, where the first 2 columns of the CSV file contain the
# keys, and the third column contains the values.


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
list(zip(keys, values))
# Out:  [('k1', 'v1'), ('k2', 'v2'), ('k3', 'v3')]

d = dict(zip(keys, values))
d
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


if __name__ == '__main__':
    pXK = load_data("ML01_pXK.csv")
    print(pXK)
    get_pX(pXK)
