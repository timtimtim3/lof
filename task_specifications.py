from algorithms.fsa import FiniteStateAutomaton

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

