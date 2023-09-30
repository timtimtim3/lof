import numpy as np
from abc import ABC, abstractmethod
from .utils import evaluate_meta_policy

class OptionBase(ABC):

    def __init__(self, env, subgoal_state):
        self.env = env
        self.s_dim = env.s_dim 
        self.states = env.states
        self.a_dim = env.action_space.n
        self.subgoal_state = subgoal_state
        self.Q = None 
        self.Ro = None
        self.To = None

    @abstractmethod
    def train(self):

        raise NotImplementedError
    

class OptionVI(OptionBase):

    def __init__(self, env, subgoal_state, gamma=1):
        super().__init__(env, subgoal_state)
        
        self.gamma = gamma

    
    def train(self):
        
        Q = np.zeros((self.s_dim, self.a_dim))
        V = np.zeros(self.s_dim)

        To = np.zeros((self.s_dim, self.s_dim)) 
        goal_idx = self.states.index(self.subgoal_state)

        To[:, goal_idx] = 1

        iters = 0

        rewards = self.env.rewards
        rewards[goal_idx] = 0
        
        while True:
            
            iters+=1

            Q_old = Q.copy()

            for sidx in range(self.s_dim):
                
                all_ns = []

                if sidx == goal_idx:
                    To[goal_idx, goal_idx] = 1 
                    continue

                for aidx in range(self.a_dim):

                    q_value = 0
                    p = prob = self.env.P[sidx, aidx, :]

                    terms = np.ones(self.s_dim)
                    terms[goal_idx] = 0

                    for nsidx in range(self.s_dim):
                       
                        prob = self.env.P[sidx, aidx, nsidx]
                        if not prob:
                            continue
                        all_ns.append(nsidx)

                        done = nsidx == goal_idx

                        q_value += prob * (self.env.reward(self.states[sidx]) + (1-done) * self.gamma * V[nsidx]) 

                    Q[sidx, aidx] = q_value
                
                To[sidx, goal_idx] = self.gamma * np.max(To[all_ns, goal_idx])
            
            V = Q.max(axis=1)

            if np.allclose(Q_old, Q):
                break 
        
        self.Q = Q
        self.Ro = V
        self.To = To

        return iters

class MetaPolicy(ABC):


    def __init__(self, env, gamma=1., record=True):

        self.env = env
        self.gamma = gamma
        self.record = record

    
    def train(self):
        pass


class MetapolicyVI(MetaPolicy):

    def __init__(self, env, eval_env, fsa, T, gamma=1., record=True, num_iters = 50):
        super().__init__(env, gamma, record)

        self.fsa = fsa 
        self.T = T
        self.num_iters = num_iters
        self.eval_env = eval_env

        if self.record:
            self.num_iterations_per_option = []

        self.options = self._learn_options()
        self.Q = None 
        self.V = None
        self.mu = None
    

    def _learn_options(self):
        """
            Automatically learns an option (with ValueIteration) per exit state.
        """

        options = []

        for exit_state in self.env.exit_states:
            option = OptionVI(self.env, exit_state, self.gamma)
            num_iters = option.train()

            options.append(option)
            
            if self.record:
                self.num_iterations_per_option.append(num_iters)

        return options


    def train_metapolicy(self):


        Q = np.zeros((len(self.fsa.states), self.env.s_dim, len(self.options)))
        V = np.zeros((len(self.fsa.states), self.env.s_dim))

        # FSA rewards (1 everywhere except at terminal)
        R = np.ones(len(self.fsa.states))
        R[-1] = 0

        for _ in range(self.num_iters):

            for oidx, option in enumerate(self.options):

                # Qo = OPTIONS[o]['Q']
                Ro, To = option.Ro - 1, option.To

                # Eq. 3 LOF paper
                rewards = R * (np.tile(Ro[:, None], [1, len(self.fsa.states)]) - 1)
                next_value = np.dot(To, V.T)

                preQ = rewards + next_value

                Q[:, :, oidx] = preQ.T

            V = Q.max(axis=2)
            preV = np.tile(V[None, ...], (len(self.fsa.states), 1, 1))
            # Multiply by T before passing to next iteration (masks value function)
            V = np.sum(self.T * preV, axis=1)

            # For each iteration, evaluate the policy
            mu_aux = Q.argmax(axis=2)
            mu = {}

            for (fidx, sidx) in np.ndindex(mu_aux.shape):
                f, s = self.fsa.states[fidx], self.env.states[sidx]
                mu[(f, s)] = mu_aux[fidx, sidx]

            success, acc_reward = self.evaluate_meta_policy(mu)
            print(_, success, acc_reward)
            


        self.Q = Q
        self.V = V

        mu_aux = Q.argmax(axis=2)
        mu = {}

        for (fidx, sidx) in np.ndindex(mu_aux.shape):
            f, s = self.fsa.states[fidx], self.env.states[sidx]
            mu[(f, s)] = mu_aux[fidx, sidx]
        
        self.mu = mu
        

    def evaluate_meta_policy(self, policy, max_steps=200):

        state = self.eval_env.reset()

        acc_reward = 0
        success = False

        for _ in range(max_steps):

            option = policy[state]
            (_, llstate) = state

            action = self.options[option].Q[self.env.states.index(llstate)].argmax()

            state, reward, done, _ = self.eval_env.step(action)
            acc_reward += reward

            if done:
                success = True
                break

        return success, acc_reward