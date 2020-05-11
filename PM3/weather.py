"""
Implementation of Weather-Umbrella HMM domain from AIMA3.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from hmm import HMM

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


class WeatherHMM(HMM):

    def __init__(self, X_domain=None, E_domain=None, trans_model=None, emission_model=None):
        self.X_domain = X_domain if X_domain else R_domain
        self.E_domain = E_domain if E_domain else U_domain
        self.T = trans_model if trans_model else default_T
        self.E = emission_model if emission_model else default_E

    def get_states(self):
        return self.X_domain

    def get_observations(self):
        return self.E_domain

    def pt(self, cur_state, next_state):
        """Return the probability of transition from current to next state"""
        return self.T[cur_state][next_state]

    def pe(self, state, obs):
        """Return the probability of the given observation when in current state"""
        return self.E[state][obs]
