"""
The main script of the exercise H: Hidden Markov Models I.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Contains functions executing various inference tasks
"""

from collections import Counter
from weather import WeatherHMM
from hmm_inference import *
from utils import normalized

# Initial distribution (can be changed)
P0 = Counter({'+rain': 0.5, '-rain': 0.5})

# State R
R_domain = {'+rain', '-rain'}
# Observation U
U_domain = {'+umb', '-umb'}

# Transition model
default_T = {
    '-rain':
        {'-rain': 0.9,
         '+rain': 0.1},
    '+rain':
        {'-rain': 0.3,
         '+rain': 0.7}
}

# Sensor (emission) model
default_E = {
    '-rain':
        {'+umb': 0.2,
         '-umb': 0.8},
    '+rain':
        {'+umb': 0.9,
         '-umb': 0.1}
}


def run_simulation(n_steps=10):
    """Simulate n_steps of WeatherHMM"""
    print('=== Simulation for {} steps'.format(n_steps))
    wtr = WeatherHMM(X_domain=R_domain, E_domain=U_domain,
                     trans_model=default_T, emission_model=default_E)
    states, observations = wtr.simulate('+rain', n_steps)
    for state, obs in zip(states, observations):
        print(state, obs)


def run_prediction(n_steps=10):
    """Predict the belief state of WeatherHMM"""
    prior = P0
    print('=== Prediction from initial state', prior)
    wtr = WeatherHMM()
    p = predict(n_steps, prior, wtr)
    for i in p:
        print(i)


def run_evidence_updates():
    """Execute several evidence updates for multiple observations"""
    print('=== Evidence updates')
    p = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', p)
    wtr = WeatherHMM()
    observations = ['+umb', '-umb', '+umb', '+umb', '+umb']
    # Initialize current beliefs with uniform distribution
    for obs in observations:
        p = normalized(update_belief_by_evidence(p, obs, wtr))
        print(obs, p)


def run_filtering():
    """Execute forward filtering algorithm for certain evidence sequence"""
    print('=== Filtering')
    wtr = WeatherHMM()
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', prior)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    f_seq = forward(prior, e_seq, wtr)
    for ft, et in zip(f_seq, e_seq):
        print(et, ft)


def run_likelihood():
    """Compare the likelihoods of two HMMs given the observation sequence"""
    print('=== Likelihood')
    wtr1 = WeatherHMM()
    E = {
        '-rain':
            {'+umb': 0.3,
             '-umb': 0.7},
        '+rain':
            {'+umb': 0.8,
             '-umb': 0.2}
    }
    wtr2 = WeatherHMM(emission_model=E)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Likelihood of HMM1:', likelihood(prior, e_seq, wtr1))
    print('Likelihood of HMM2:', likelihood(prior, e_seq, wtr2))


def update_belief_by_time_step_test():
    wtr = WeatherHMM()
    B = Counter({'+rain': 0.1, '-rain': 0.9})
    Counter1 = update_belief_by_time_step(B, wtr)
    Counter2 = Counter({'+rain': 0.16, '-rain': 0.84})
    print("Counter1", Counter1)
    print("Counter2", Counter2)
    print("Counter1 and Counter2 are the same when correct.")


def update_belief_by_evidence_test():
    wtr = WeatherHMM()
    B = Counter({'+rain': 0.5, '-rain': 0.5})
    B_new = update_belief_by_evidence(B, '-umb', wtr)
    print("Counter1", normalized(B_new))
    print("Counter2", Counter(
        {'+rain': 0.11111111111111112, '-rain': 0.888888888888889}))
    print("Counter1 and Counter2 are the same when correct.")


def forwart1_test():
    wtr = WeatherHMM()
    f = Counter({'+rain': 0.5, '-rain': 0.5})
    f_new = forward1(f, '-umb', wtr)
    print("f_new", f_new)
    print("f correct answer", Counter({'+rain': 0.04, '-rain': 0.48}))
    print("f correct ans norm", normalized(
        Counter({'+rain': 0.04, '-rain': 0.48})))


if __name__ == '__main__':
    print('All functions are turned on.')

    run_simulation()
    run_prediction()
    run_evidence_updates()
    run_filtering()
    #run_likelihood()


#
#   scroll
#
