import numpy as np


class ValueIteration:

    def __init__(self, env, gamma=0.99):

        self.env = env
        self.s_dim = len(env.state_to_coords)
        self.a_dim =  env.action_space.n
        self.states = list(env.coords_to_state.keys())
        self.gamma = gamma

    
    def train(self, goal_state):

        
        Q = np.zeros((self.s_dim, self.a_dim))
        V = np.zeros(self.s_dim)

        total_iterations = 0

        To = np.zeros((self.s_dim, self.s_dim)) 
        goal_idx = self.states.index(goal_state)

        To[goal_idx, goal_idx] = 1

        iters = 0
        
        while True:
            iters+=1

            total_iterations += 1
            Q_old = Q.copy()
            V_old = V.copy()

            for sidx in range(self.s_dim):
                
                all_ns = []

                for aidx in range(self.a_dim):

                    q_value = 0

                    for nsidx in range(self.s_dim):
                       
                        prob = self.env.P[sidx, aidx, nsidx]
                        if not prob:
                            continue
                        all_ns.append(nsidx)

                        done = nsidx == goal_idx

                        q_value += prob * (self.env.reward(self.states[sidx]) + (1-done) * self.gamma * V_old[nsidx]) 
                        if nsidx == goal_idx:
                            To[sidx, nsidx] = self.gamma 

                    Q[sidx, aidx] = q_value
                    V = Q.max(axis=1)
                    To[sidx, goal_idx] = self.gamma * np.max(To[all_ns, goal_idx])

            if np.allclose(Q_old, Q):
                break 
        

        return Q, V, To
 