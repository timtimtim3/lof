import gym
import numpy as np

class DeliveryAutomatonEnv(gym.Env):


    def __init__(self, env, fsa, fsa_init_state, T):

        self.env = env 
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.T = T

    def get_state(self):

        return (self.fsa_state, tuple(self.env.state))

    def reset(self):

        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset())
        
        return (self.fsa_state, self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        self.env.step(action) 
        state = self.env.state
        state_index = self.env.states.index(state)

        fsa_state_index = self.fsa.states.index(self.fsa_state)

        next_fsa_state_idxs = np.where(self.T[fsa_state_index, :, state_index] == 1)[0]

        if len(next_fsa_state_idxs) == 0:
            return (self.fsa_state, state), -100, True, {}
        else: 
            next_fsa_state_index = next_fsa_state_idxs.item()
        
        self.fsa_state = self.fsa.states[next_fsa_state_index]

        done = self.fsa.is_terminal(self.fsa_state)

        return (self.fsa_state, state), -1, done, {}



