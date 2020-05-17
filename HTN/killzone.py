# Author:  Radek Marik, RMarik@gmail.com
# Created: April 22, 2017
# Purpose: Pyhop - killZone domain example

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

PYHOP_VERSION = 1  # 1.2.2 (official) or 2.0

if PYHOP_VERSION == 2:
    # Pyhop2.0
    from pyhop import helpers, hop
else:
    from pyhop import pyhop as plan
    from pyhop import State, declare_operators, declare_methods, print_state, find_if

    Goal = State
    print_goal = print_state
Set_C = set

"""The killZone operators use three state variables:
- field = a network of position links, it is static.
- monsters[m] = the static position of the monster 'm'
- medic = which soldier has medical skills, it is static
- soldiers[s] = the current position of the soldier 's'
- heatlhy[s] = it is True if the soldier 's' is healthy, otherwise False
"""

"""We do not need any goal in this game."""

verbose = 1

# ===================== Support Structures ==============================

links = [('V11', 'V21'), ('V12', 'V22'), ('V13', 'V23'), ('V14', 'V24'), ('V15', 'V25'), ('V16', 'V26'),
         ('V21', 'V31'), ('V22', 'V32'), ('V23', 'V33'), ('V24', 'V34'), ('V25', 'V35'), ('V26', 'V36'),
         ('V31', 'V41'), ('V32', 'V42'), ('V33', 'V43'), ('V34', 'V44'), ('V35', 'V45'), ('V36', 'V46'),
         ('V41', 'V51'), ('V42', 'V52'), ('V43', 'V53'), ('V44', 'V54'), ('V45', 'V55'), ('V46', 'V56'),
         ('V11', 'V12'), ('V12', 'V13'), ('V13', 'V14'), ('V14', 'V15'), ('V15', 'V16'), ('V21', 'V22'),
         ('V22', 'V23'), ('V23', 'V24'), ('V24', 'V25'), ('V25', 'V26'), ('V31', 'V32'), ('V32', 'V33'),
         ('V33', 'V34'), ('V34', 'V35'), ('V35', 'V36'), ('V41', 'V42'), ('V42', 'V43'), ('V43', 'V44'),
         ('V44', 'V45'), ('V45', 'V46'), ('V51', 'V52'), ('V52', 'V53'), ('V53', 'V54'), ('V54', 'V55'),
         ('V55', 'V56')
         ]

print("Define the initial state1:")
state1 = State('state1')
state1.field = {}
for pX, pY in links:
    state1.field.setdefault(pX, Set_C()).add(pY)
    state1.field.setdefault(pY, Set_C()).add(pX)
state1.soldiers = {'Rambo': 'V11', 'Metris': 'V12', 'Medic': 'V12'}
state1.medic = 'Medic'
state1.monsters = {'Alien': 'V15', 'Klingon': 'V22', 'Predator': 'V31', 'Godzilla': 'V53', 'Jozin': 'V26'}
state1.healthy = {s: True for s in state1.soldiers}
print_state(state1)
print('')


# ===================== Support Functions ===============================

def HasSupport(state, soldier):
    soldier_pos = state.soldiers[soldier]
    for suppId, supp_pos in state.soldiers.items():
        if suppId == soldier:
            continue
        if soldier_pos in state.field[supp_pos]:
            return True
    return False


def ShortestPath(state, from_loc, to_loc):

    if 0: print('ShortestPath.start', from_loc, to_loc)
    queue = [from_loc]
    backward = {}
    while queue:
        pos = queue.pop(0)
        if pos not in backward:
            backward[pos] = None  # created

        if 0:
            print('queue', pos, queue, backward)

        for next_pos in state.field[pos]:
            if next_pos in backward:
                continue
            backward[next_pos] = pos

            if next_pos == to_loc:
                path = []

                while next_pos is not None:
                    path.append(next_pos)
                    next_pos = backward[next_pos]

                path.reverse()

                return path
            queue.append(next_pos)

    raise Exception('This is a failure!!! Check the input data')


def GetNearestSoldierPath(state, loc):
    s_paths = {soldier: ShortestPath(state, soldierPos, loc)
              for soldier, soldierPos in state.soldiers.items()
              if soldierPos != loc}

    soldier = min(s_paths, key=lambda s: len(s_paths[s]))
    if 0:
        print('GetNearestSoldierPath', soldier)
    return soldier, s_paths[soldier]


# Operators
# =========


# Methods
# =======


def test_plan():
    # ==================
    mission_plan = plan(state1, [('killMonsters',)], verbose=verbose)
    for action in mission_plan:
        print('action:', action)


if __name__ == '__main__':

    test_plan()

