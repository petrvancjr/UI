"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized


def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    # Your code here
    raise NotImplementedError('You must implement update_belief_by_time_step()')
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
    # Your code here
    raise NotImplementedError('You must implement predict()')        
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
    # Your code here
    raise NotImplementedError('You must implement update_belief_by_evidence()')
    return cur_B


def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    # Your code here
    raise NotImplementedError('You must implement forward1()')
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
    # Your code here
    raise NotImplementedError('You must implement forward()')
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