"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized

import numpy as np


def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()

    X = hmm.get_states()  # Possible current states
    for x_ in hmm.get_states():
        for X_ in X:
            # B(X) += P(X,x')*B(x')
            cur_B[X_] += hmm.pt(x_, X_) * prev_B[x_]

    return cur_B


def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B = prior  # This shall be iteratively updated
    Bs = []    # This shall be a collection of Bs over time steps

    for i in range(n_steps):
        cur_B = update_belief_by_time_step(B, hmm)
        Bs.append(cur_B)
        B = cur_B
    return Bs


def update_belief_by_evidence(prev_B, e, hmm):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :return: Counter, current (updated) belief distribution over states
    """
    # Create a new copy of the current belief state
    cur_B = Counter(prev_B)

    X = hmm.get_states()
    for X_ in X:
        cur_B[X_] = hmm.pe(X_, e) * prev_B[X_]

    return cur_B


def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    alpha = 1
    #single only single update

    cur_f = Counter()
    upd = update_belief_by_time_step(prev_f, hmm)
    for X_ in hmm.get_states():
        #print("cur_e", cur_e, "prev_f[X_]", prev_f[X_])
        cur_f[X_] = alpha * hmm.pe(X_, cur_e) * upd[X_]

    return cur_f


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, i.e., estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice

    for e_kus in e_seq:

        f = normalized(forward1(f, e_kus, hmm))
        fs.append(f)

    return fs


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    # Your code here
    raise NotImplementedError('You must implement likelihood()')
    return lhood


def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()

    for X_ in hmm.get_states():
        cur_b[X_] = 0.0  # Probably unnecessary
        for x_ in hmm.get_states():
            cur_b[X_] += hmm.pe(x_, next_e) * hmm.pt(X_, x_) * next_b[x_]

    return cur_b


def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []   # Smoothed belief distributions

    t = len(e_seq)
    f = []
    fi = priors
    f.append(priors)
    for ei in e_seq:
        fi = normalized(forward1(fi, ei, hmm))
        f.append(fi)
    #f = forward(priors, e_seq, hmm)
    # note f dont have prior inside

    b = Counter()
    for X_ in hmm.get_states():
        b[X_] = 1
    for n in range(t, 0, -1):
        #print("f[n]", f[n], "b", b)
        si = normalized(CounterDictCross(f[n], b, hmm))
        b = backward1(b, e_seq[n-1], hmm)

        se.insert(0, si)

    return (se)


def CounterDictCross(f, b, hmm):
    c = Counter()
    for X_ in hmm.get_states():
        c[X_] = (f[X_] * b[X_])
    return c


def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()   # Current (updated) max message
    predecessors = {}   # The best of previous states for each current state

    # Nested function - only changing the sum to argmax or max?
    def update_belief_by_time_step_max(prev_B, hmm):
        cur_B = Counter()
        predecessors = {}

        X = hmm.get_states()  # Possible current states
        for n, x_ in enumerate(hmm.get_states()):
            for m, X_ in enumerate(X):
                if cur_B[X_] < (hmm.pt(x_, X_) * prev_B[x_]):
                    cur_B[X_] = hmm.pt(x_, X_) * prev_B[x_]
                    predecessors[X_] = x_

        return cur_B, predecessors

    # Iterate over current and previous state
    maxs, predecessors = update_belief_by_time_step_max(prev_m, hmm)
    for X_ in hmm.get_states():

        cur_m[X_] = hmm.pe(X_, cur_e) * maxs[X_]

    return cur_m, predecessors


def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []      # Sequence of max messages
    m = [priors]
    prev_e = None
    for i, cur_e in enumerate(e_seq):

        if i == 0:
            m.append(forward1(m[i], cur_e, hmm))
            # Init
            X_ = {'+rain': '+rain'}
            prev_e = X_['+rain']
        else:
            tmp, X_ = viterbi1(m[i], cur_e, hmm)
            m.append(tmp)
            prev_e = X_[prev_e]

        ms.append(m[i+1])
        ml_seq.append(X_[prev_e])

    return ml_seq, ms


#
#   Scroll
#
