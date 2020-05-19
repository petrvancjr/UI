# Author:  Radek Marik, RMarik@gmail.com
# Created: April 22, 2017
# Purpose: Pyhop - blocks domain example

# Copyright (c) 2017, Radek Marik, FEE CTU, Prague
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of FEE CTU nor the
#       names of contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

# import os.path

PYHOP_VERSION = 1  # 1.2.2 (official) or 2.0

if PYHOP_VERSION == 2:
    from pyhop import helpers, hop
else:
    import pyhop
    from pyhop import pyhop as plan

    Goal = pyhop.State
    print_goal = pyhop.print_state

Set_C = set

DEBUG = True

"""The blocks-world operators use three state variables:
- pos[b] = block b's position, which may be 'table', 'hand', or another block.
- clear[b] = False if a block is on b or the hand is holding b, else True.
- holding = name of the block being held, or False if the hand is empty.

A goal is a collection of some (but not necessarily all) of the state variables
and their desired values. Below, both goal1a and goal1b specify c on b, and b
on a. The difference is that goal1a also specifies that a is on table and the
hand is empty.
"""

VERBOSE = 1

# ===================== Support Structures ==============================
print("Define the initial state1: a on b, b on table, c on table")
state1 = pyhop.State('state1')
state1.pos = {'a': 'b', 'b': 'table', 'c': 'table'}
#state1.pos = {'a': 'b', 'b': 'table', 'c': 'a'}
state1.clear = {'c': True, 'b': False, 'a': True}
state1.holding = False
pyhop.print_state(state1)
print('')


print("Define the goal goal1a:\n=======================")
goal_a = Goal('goal1a')
goal_a.pos = {'c': 'b', 'b': 'a', 'a': 'table'}
goal_a.clear = {'c': True, 'b': False, 'a': False}
goal_a.holding = False
print_goal(goal_a)


# ===================== Support Functions ===============================
def block_is_done(b1, state, goal, done_state):
    if b1 == done_state:
        return True

    if b1 in goal.pos and goal.pos[b1] != state.pos[b1]:
        return False

    if state.pos[b1] == done_state:
        return True

    return block_is_done(state.pos[b1], state, goal, done_state)


def all_blocks(state):
    """Returns all block ids."""
    return state.clear.keys()


def block_status(block_id, state, goal, done_state):
    """
    A helper function used in the methods' preconditions.
    """
    if block_is_done(block_id, state, goal, done_state):
        return 'done'

    elif not state.clear[block_id]:
        return 'inaccessible'

    elif not (block_id in goal.pos) or goal.pos[block_id] == done_state:
        return 'move-to-table'

    elif block_is_done(goal.pos[block_id], state, goal, done_state) and \
            state.clear[goal.pos[block_id]]:
        return 'move-to-block'

    else:
        return 'waiting'


"""
Operators
"""


def pickup(state, x):  # I did not define, param o when calling
    if state.pos[x] == 'table' and state.holding == False and state.clear[x] == True:
        state.pos[x] = 'hand'
        state.holding = True
        state.clear[x] = False
        return state
    return False


def putdown(state, x):  # I did not define, param p when calling
    if state.pos[x] == 'hand' and state.holding == True:
        state.pos[x] = 'table'
        state.holding = False
        state.clear[x] = True
        return state
    return False


def stack(state, x, y):
    if state.pos[x] == 'hand':
        if state.holding == True:  # Check x
            if state.clear[y] == True:  # Check y on top
                state.pos[x] = y
                state.clear[x] = True
                state.clear[y] = False
                state.holding = False
                return state
    return False


def unstack(state, x, y):
    if state.pos[x] == y and state.holding == False:  # Check x
        if state.clear[x] == True:  # Check y on top
            state.pos[x] = 'hand'
            state.clear[x] = False
            state.clear[y] = True
            state.holding = True
            return state
    return False


pyhop.declare_operators(pickup, unstack, putdown, stack)
pyhop.print_operators()

"""
Methods
"""


def move_one_block(state, x, y):  # Moving x -> y
    path = []
    if state.clear[x] == False:  # Preconditions
        return False
    if get_block(state, x):  # 1. Pick up
        path.append(get_block(state, x))
    else:
        return False
    if put_block(state, x, y):  # 2. Put Down
        path.append(put_block(state, x, y))
    else:
        return False
    return path


def get_block(state, x):
    if state.pos[x] == 'table':
        return ('pickup', x)
    elif state.pos[x] == 'hand':
        return False
    else:  # position is on some block
        y = state.pos[x]
        return ('unstack', x, y)


def put_block(state, x, y):
    if y == 'table':
        return ('putdown', x)
    elif y == 'hand':
        return False
    else:  # Stack on something
        state = stack(state, x, y)
        return ('stack', x, y)


def move_blocks(state, goal):
    blocks = all_blocks(state)
    # Make a stack
    goalStack = []
    for block in blocks:
        if goal.clear[block] == True:  # the top block
            while block != 'table':
                goalStack.insert(0, block)
                block = goal.pos[block]

    topBlocks = []
    for block in blocks:
        if state.clear[block] == True:  # the top block
            topBlocks.append(block)

    stackString = []
    for kominek in topBlocks:
        while state.pos[kominek] != 'table':
            stackString.append(('move_one_block', kominek, 'table'))
            kominek = state.pos[kominek]

    stackOn = 'table'  # Init -> lowest block goes onto table
    for n, x in enumerate(goalStack):
        if n == 0:
            stackOn = x
        else:
            stackString.append(('pickup', x))
            stackString.append(('stack', x, stackOn))
            stackOn = x
    return stackString


pyhop.declare_methods('move_one_block', move_one_block)
pyhop.declare_methods('get_block', get_block)  # either pickup or unstack
pyhop.declare_methods('put_block', put_block)  # either putdown or stack
pyhop.declare_methods('move_blocks', move_blocks)

print('\n')
pyhop.print_methods()


if __name__ == '__main__':

    def test_operators_methods():
        # =============================

        print('- these should fail:\n=======================')
        plan(state1, [('pickup', 'b')], verbose=VERBOSE)
        plan(state1, [('pickup', 'a')], verbose=VERBOSE)
        plan(state1, [('pickup', 'c'), ('stack', 'c', 'b')], verbose=VERBOSE)

        print('- these should pass:\n=======================')
        plan(state1, [('pickup', 'c')], verbose=VERBOSE)

        plan(state1, [('pickup', 'c'), ('putdown', 'c')], verbose=VERBOSE)
        plan(state1, [('unstack', 'a', 'b')], verbose=VERBOSE)
        plan(state1, [('pickup', 'c'), ('stack', 'c', 'a')], verbose=VERBOSE)

    def test_method():
        print('\n')
        plan(state1, [('move_one_block', 'c', 'a')], verbose=VERBOSE)
        plan(state1, [('move_one_block', 'a', 'table')], verbose=VERBOSE)

    def test_plan():
        print('\n')
        a_plan = plan(state1, [('move_blocks', goal_a)], verbose=VERBOSE)

        for action in a_plan:
            print('action:', action)

    test_operators_methods()
    test_method()
    test_plan()
