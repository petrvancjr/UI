"""
The "travel from home to the park" example from my lectures.
Author: Dana Nau <nau@cs.umd.edu>, May 31, 2013
This file should work correctly in both Python 2.7 and Python 3.2.

Travel from home to park:
operators:
- walk
- call_taxi
- ride_taxi
- pay_driver
+ ride_bike

methods:
- travel_by_foot
- travel_by_taxi
+ travel_by_bike

Edit 17 May 2020: Added bike -> Constraints: Bike must be at place, Bike cannot be stolen
"""

import pyhop


# ============== Supporting function =============
def taxi_rate(dist):
    return 1.5 + 0.5 * dist


# ============== operators ===================

def walk(state, a, x, y):
    if state.loc[a] == x:
        state.loc[a] = y
        return state
    else:
        return False


def call_taxi(state, a, x):
    state.loc['taxi'] = x
    return state


def ride_taxi(state, a, x, y):
    if state.loc['taxi'] == x and state.loc[a] == x:
        state.loc['taxi'] = y
        state.loc[a] = y
        state.owe[a] = taxi_rate(state.dist[x][y])
        return state
    else:
        return False


def pay_driver(state, a):
    if state.cash[a] >= state.owe[a]:
        state.cash[a] = state.cash[a] - state.owe[a]
        state.owe[a] = 0
        return state
    else:
        return False


def ride_bike(state, a, x, y):
    if state.loc['bike'] == x and state.loc[a] == x:
        state.loc[a] = y
        state.loc['bike'] = y
        return state
    else:
        return False


pyhop.declare_operators(walk, call_taxi, ride_taxi, pay_driver, ride_bike)
print('')
pyhop.print_operators()


# ============== methods ===================

def travel_by_foot(state, a, x, y):
    if state.dist[x][y] <= 2:
        return [('walk', a, x, y)]
    return False


def travel_by_taxi(state, a, x, y):
    if state.cash[a] >= taxi_rate(state.dist[x][y]):
        return [('call_taxi', a, x), ('ride_taxi', a, x, y), ('pay_driver', a)]
    return False


def travel_by_bike(state, a, x, y):
    if state.dist[x][y] < 10:
        return [('ride_bike', a, x, y)]
    return False


pyhop.declare_methods('travel', travel_by_foot, travel_by_bike, travel_by_taxi)
print('')
pyhop.print_methods()

state1 = pyhop.State('state1')
state1.loc = {'me': 'home', 'bike': 'home'}
state1.cash = {'me': 20}
state1.owe = {'me': 0}
state1.dist = {'home': {'park': 8}, 'park': {'home': 8}}

if __name__ == '__main__':

    # print("- If verbose=0 (the default), Pyhop returns the solution but prints nothing.\n")
    # pyhop.pyhop(state1, [('travel', 'me', 'home', 'park')])

    print('- If verbose=1, Pyhop prints the problem and solution, and returns the solution:')
    pyhop.pyhop(state1, [('travel', 'me', 'home', 'park')], verbose=1)

    #print('- If verbose=2, Pyhop also prints a note at each recursive call:')
    #pyhop.pyhop(state1, [('travel', 'me', 'home', 'park')], verbose=2)

    #print('- If verbose=3, Pyhop also prints the intermediate states:')
    #pyhop.pyhop(state1, [('travel', 'me', 'home', 'park')], verbose=3)
