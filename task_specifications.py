from algorithms.fsa import FiniteStateAutomaton
import numpy as np

def fsa_delivery_mini1():

    symbols_to_phi = {"A": [0], 
                      "B":[1], 
                      "H": [2]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "H")

    return fsa

def fsa_delivery_mini2():

    symbols_to_phi = {"A": [0], 
                      "H": [2]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "H")

    return fsa

def fsa_delivery1(env):


    symbols_to_phi = {"A": [0], 
                      "B": [1], 
                      "C": [2], 
                      "H": [3]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    exit_states_idxs = list(map(lambda x: env.states.index(x), env.exit_states)) 
    obstacles_idxs = list(map(lambda x: env.states.index(x), env.obstacles))
    
    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    T[0, 0, :] = 1
    T[0, 0, obstacles_idxs] = 0
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 

    T[1, 1, :] = 1
    T[1, 1, obstacles_idxs] = 0
    T[1, 1, exit_states_idxs[1]] = 0
    T[1, 2, exit_states_idxs[1]] = 1

    T[2, 2, :] = 1
    T[2, 2, obstacles_idxs] = 0
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 3, exit_states_idxs[2]] = 1

    T[3, 3, :] = 1
    T[3, 3, obstacles_idxs] = 0
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1

    return fsa, T


if __name__ == "__main__":

    fsa, _ = fsa_delivery_mini2()

    M = fsa.get_transition_matrix()

    print(M)