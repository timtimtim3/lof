import numpy as np
from abc import ABC, abstractmethod
import time
import pickle as pkl
import os
from time import sleep

def __init__():
    pass

class OptionBase(ABC):

    def __init__(self, env, subgoal_state, gamma):
        self.env = env
        self.s_dim = env.s_dim 
        self.states = env.states
        self.a_dim = env.action_space.n
        self.subgoal_state = subgoal_state
        self.subgoal_idx = env.states.index(subgoal_state)
        self.gamma = gamma
        self.Q = np.zeros((self.s_dim, self.a_dim))
        self.Ro = np.zeros(self.s_dim)
        self.To = np.zeros((self.s_dim, self.s_dim))


    def save(self, path: str):
        
        pkl.dump(self.Q, open(os.path.join(f"{path}", "Q.pkl"), "wb"))
        pkl.dump(self.Ro, open(os.path.join(f"{path}", "Ro.pkl"), "wb"))
        pkl.dump(self.To, open(os.path.join(f"{path}", "To.pkl"), "wb"))
    

class OptionVI(OptionBase):

    def __init__(self, env, subgoal_state, gamma=1):
        super().__init__(env, subgoal_state, gamma)
        
    
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
                    continue

                for aidx in range(self.a_dim):

                    q_value = 0

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
    
class OptionQLearning(OptionBase):

    def __init__(self, env, subgoal_state, gamma=1, alpha=0.2):
        super().__init__(env, subgoal_state, gamma)
        
        self.alpha = alpha
        self.To = np.zeros((self.s_dim, self.s_dim))
        self.To[self.subgoal_idx, self.subgoal_idx] = 1

    
    def update_qfunction(self, state, action, next_state, reward):

        if state != self.subgoal_state:

            done = next_state == self.subgoal_state

            state_idx = self.env.states.index(state)
            next_state_idx =  self.env.states.index(next_state)
            subgoal_idx = self.env.states.index(self.subgoal_state)

            update = reward + self.gamma * (1 - done) * self.Q[next_state_idx, :].max() - self.Q[state_idx, action]
            self.Q[state_idx, action] += self.alpha * update
            self.Ro[state_idx] = self.Q[state_idx, :].max()
            u1, u2 = self.To[state_idx, subgoal_idx], self.gamma * self.To[next_state_idx, subgoal_idx]
            self.To[state_idx, subgoal_idx] = np.max([u1, u2])


class MetaPolicy(ABC):


    def __init__(self, env, eval_env, fsa, T, gamma=1., num_iters = 50, writer=None):

        self.env = env
        self.eval_env  = eval_env
        self.fsa = fsa
        self.T = T
        self.gamma = gamma
        self.writer = writer
        self.num_iters = num_iters

        self.Q = None
        self.V = None

    @abstractmethod
    def _learn_options(self):
        raise NotImplementedError 

    # def train(self):
    #     raise NotImplementedError 
    
    @abstractmethod
    def evaluate_metapolicy(self, policy, log=False, max_steps=100, max_steps_option=30):
        raise NotImplementedError
    
    def save(self, path:str):
        
        for i, option in enumerate(self.options):

            fullpath = os.path.join(path, f"options/option{i}/")
            os.makedirs(fullpath)
            option.save(fullpath)
    
        pkl.dump(self.Q, open(os.path.join(path,"Q.pkl"), "wb"))
        pkl.dump(self.mu, open(os.path.join(path,"metapolicy.pkl"), "wb"))
    

    def train_metapolicy(self, record=False, iters = None):

        if self.Q is None and self.V is None:
            self.Q = np.zeros((len(self.fsa.states), self.env.s_dim, len(self.options)))
            self.V = np.zeros((len(self.fsa.states), self.env.s_dim))

        # FSA rewards (1 everywhere except at terminal)
        R = np.ones(len(self.fsa.states))
        terminals = [self.fsa.is_terminal(s) for s in self.fsa.states]
        R[np.where(terminals)] = 0

        times = [0]

        num_iters = self.num_iters if iters is None else iters

        for j in range(self.num_iters):

            iter_start = time.time() if record else None

            for oidx, option in enumerate(self.options):

                # Qo = OPTIONS[o]['Q']
                Ro, To = option.Ro - 1, option.To

                # Eq. 3 LOF paper
                rewards = R * (np.tile(Ro[:, None], [1, len(self.fsa.states)]) - 1)
                next_value = np.dot(To, self.V.T)

                preQ = rewards + next_value

                self.Q[:, :, oidx] = preQ.T
            
            self.V = self.Q.max(axis=2)
            preV = np.tile(self.V[None, ...], (len(self.fsa.states), 1, 1))
            # Multiply by T before passing to next iteration (masks value function)
            self.V = np.sum(self.T * preV, axis=1)

            # For each iteration, evaluate the policy
            mu_aux = self.Q.argmax(axis=2)
            mu = {}

            elapsed_iter = time.time() - iter_start if record else None

            for (fidx, sidx) in np.ndindex(mu_aux.shape):
                f, s = self.fsa.states[fidx], self.env.states[sidx]
                mu[(f, s)] = mu_aux[fidx, sidx]

            times.append(elapsed_iter)
            
            if record:
    
                success, acc_reward = self.evaluate_metapolicy(mu)
                self.writer.add_scalar("metrics/evaluation/success", int(success), j)
                self.writer.add_scalar("metrics/evaluation/acc_reward", acc_reward, j)
                self.writer.add_scalar("metrics/evaluation/iter", j, j)
                self.writer.add_scalar("metrics/evaluation/time", np.sum(times), j)


        mu_aux = self.Q.argmax(axis=2)
        mu = {}

        for (fidx, sidx) in np.ndindex(mu_aux.shape):
            f, s = self.fsa.states[fidx], self.env.states[sidx]
            mu[(f, s)] = mu_aux[fidx, sidx]
        
        self.mu = mu
        return mu
    
    def evaluate_metapolicy(self, policy, max_steps=100, max_steps_option=40):

        acc_reward, success = 0, False
        num_steps = 0

        (f_state, state) = self.eval_env.reset()

        options_used = 0

        # print("Init:")
        while num_steps < max_steps:

            option = policy[(f_state, state)]
            options_used+=1
            
            old_f_state = f_state
            steps_in_option = 0
            done = False

            while steps_in_option < max_steps_option and state != self.options[option].subgoal_state:

                action = self.options[option].Q[self.env.states.index(state)].argmax()

                # print(f"State: {state}, Option: {option}, Action: {action}")
                sleep(0.1)

                (f_state, state), reward, done, _ = self.eval_env.step(action)

                
                num_steps+=1
                acc_reward += reward
                steps_in_option+=1
                
                if done:
                    break

            if done:
                success = self.fsa.is_terminal(f_state)
                break
        
        return success, acc_reward

# ANCHOR: metapolicy
class MetaPolicyVI(MetaPolicy):

    def __init__(self, env, eval_env, fsa, T, gamma=1., num_iters = 50, writer=None):
        super().__init__(env, eval_env, fsa, T, gamma, num_iters, writer)

        self.num_iters = num_iters
        self.eval_env = eval_env

        self.options = self._learn_options()
        self.Q = None 
        self.V = None
        self.mu = None
    

    def _learn_options(self):
        
        """
            Automatically learns an option (with ValueIteration) per exit state.
        """

        options = []

        for i, exit_state in enumerate(self.env.exit_states):

            option = OptionVI(self.env, exit_state, self.gamma)
            num_iters = option.train()

            options.append(option)

            self.writer.add_scalar(f"options/option{i}", num_iters)

        return options


class MetaPolicyQLearning(MetaPolicy):

    def __init__(self, env, eval_env, fsa, T, gamma=1., num_iters = 50, writer=None, alpha=0.2, epsilon=0.3, num_episodes=200, episode_length=100, eval_freq=50):
        
        super().__init__(env, eval_env, fsa, T, gamma, num_iters, writer)
        
        self.alpha = alpha

        self.options = []
        self.epsilon = epsilon
        self.num_episodes = num_episodes 
        self.episode_length = episode_length
        self.eval_freq = eval_freq

        exit_states_idxs = list(map(self.env.states.index, self.env.exit_states))
        self.exit_states_idx = exit_states_idxs

        for subgoal in env.exit_states:

            option = OptionQLearning(self.env, subgoal, self.gamma, self.alpha)
            self.options.append(option)


    def get_epsilon_greedy_action(self, option_idx, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            state_idx = self.options[option_idx].states.index(state)
            return self.options[option_idx].Q[state_idx, :].argmax()

    def _learn_options(self):
        
        num_options = len(self.options)

        self.env.reset()
        total_steps = 0

        for i in range(self.num_episodes):

            self.env.random_reset()

            # Use one option at a time
            option_idx = i % num_options    
            
            num_steps = 0

            while num_steps  < self.episode_length:
                
                num_steps += 1
                total_steps += 1

                current_state  = self.env.state
                action = self.get_epsilon_greedy_action(option_idx, current_state)
                _, r, _, _ = self.env.step(action)
                next_state = self.env.state

                # Intra option learning
                for option in self.options:
                    option.update_qfunction(current_state, action, next_state, r)

                if total_steps % self.eval_freq == 0:
                    # Log in tensorboard the performance during training
                    metapolicy = self.train_metapolicy(False)
                    success, reward = self.evaluate_metapolicy(metapolicy)
                    self.writer.add_scalar("learning/success", int(success), total_steps)
                    self.writer.add_scalar("learning/reward", reward, total_steps)
                    self.writer.add_scalar("learning/episode", i, total_steps)
            

        self.env.reset()

    

    
   