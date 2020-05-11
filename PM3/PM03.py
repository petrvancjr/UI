"""
Hidden Markov Models II.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Contains functions executing various inference tasks
"""

from itertools import product

from hmm_inference import *
from weather import WeatherHMM


def run_backward1():
    """Try a single step of backward algorithm"""
    print('Backward1')
    wtr = WeatherHMM()
    b = Counter({'+rain': 0.6, '-rain': 0.7})
    e = '+umb'
    print('Initial backward message:', b)
    print('Observation:', e)
    print('Updated backward message:', backward1(b, e, wtr))


def run_fb():
    """Compare the results of filtering and smoothing"""
    print('Comparison of filtering and smoothing')
    wtr = WeatherHMM()
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', prior)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    f_seq = forward(prior, e_seq, wtr)          # Filtered beliefs
    s_seq = forwardbackward(prior, e_seq, wtr)  # Smoothed beliefs
    for t, (et, ft, st) in enumerate(zip(e_seq, f_seq, s_seq)):
        print('Observation at time', t+1,':', et)
        print('Filtered:', ft)
        print('Smoothed:', st)


def run_viterbi1():
    """Try a single step of viterbi algorithm"""
    print('Viterbi1')
    wtr = WeatherHMM()
    m = Counter({'+rain': 0.45, '-rain': 0.1})
    e = '+umb'
    print('Initial max message:', m)
    print('Observation:', e)
    m, pred = viterbi1(m, e, wtr)
    print('Updated max message:', m)
    print('Best predecessors', pred)


def run_viterbi():
    print('Viterbi')
    wtr = WeatherHMM()
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', prior)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    seq, ms = viterbi(prior, e_seq, wtr)
    for e, m in zip(e_seq, ms):
        print(e, 'Max msg:', m)
    print('ML seq of states:', seq)


if __name__ == '__main__':
    print('Comment/uncomment individual run_* functions in the main section as needed.')
    #run_backward1()
    run_fb()
    #run_viterbi1()
    #run_viterbi()
