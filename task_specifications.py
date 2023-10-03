from algorithms.fsa import FiniteStateAutomaton
import numpy as np

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

def fsa_delivery2(env):


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
    fsa.add_transition("u0", "u2", "B")
    fsa.add_transition("u1", "u3", "C")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    exit_states_idxs = list(map(lambda x: env.states.index(x), env.exit_states)) 
    obstacles_idxs = list(map(lambda x: env.states.index(x), env.obstacles))
    
    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via A) or u2 (via B)
    T[0, 0, :] = 1
    T[0, 0, obstacles_idxs] = 0
    # If it goes to A or B, it transitions to a new F-state 
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 
    T[0, 2, exit_states_idxs[1]] = 1 


    # From u1 to u3
    T[1, 1, :] = 1
    T[1, 1, obstacles_idxs] = 0
    T[1, 1, exit_states_idxs[2]] = 0
    T[1, 3, exit_states_idxs[2]] = 1

    # From u2 to u3
    T[2, 2, :] = 1
    T[2, 2, obstacles_idxs] = 0
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 3, exit_states_idxs[2]] = 1
    
    # From u2 to u4
    T[3, 3, :] = 1
    T[3, 3, obstacles_idxs] = 0
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1

    return fsa, T



if __name__ == "__main__":

    fsa, _ = fsa_delivery1()

    M = fsa.get_transition_matrix()

    print(M)