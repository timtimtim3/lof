
import networkx as nx

class FiniteStateAutomaton:

    def __init__(self, symbols_to_phi) -> None:
        self.graph =  nx.DiGraph()
        self.num_states = 0
        self.states = []
        self.symbols_to_phi = symbols_to_phi

    def add_state(self, node_name):
        self.graph.add_node(node_name) 
        self.num_states += 1
        self.states.append(node_name)

    def add_transition(self, src_state, dst_state, label):
        self.graph.add_edge(src_state, dst_state, predicate = label)

    def in_transitions(self, node):
        return list(self.graph.in_edges(node))
    
    def get_predicate(self, edge):
        predicates = nx.get_edge_attributes(self.graph, 'predicate')
        return predicates[edge]
    
    def get_exit_state_idx(self, edge):
        predicate = self.get_predicate(edge)
        return self.symbols_to_phi[predicate]

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def is_terminal(self, node):
        return len(list(self.graph.neighbors(node))) < 1



