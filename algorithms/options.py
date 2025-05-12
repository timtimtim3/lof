from fsa.fsa import FiniteStateAutomaton
from abc import ABC, abstractmethod
from typing import Optional

import pickle as pkl
import numpy as np
import wandb as wb
import time
import os


class OptionBase(ABC):

    def __init__(self, env, 
                subgoal_index:int,
                gamma: float):
        
        self.env = env
        self.n_states = env.s_dim 
        self.n_actions = env.action_space.n
        self.subgoal_state = env.state_to_coords[subgoal_index]
        self.subgoal_idx = subgoal_index
        self.gamma = gamma
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.Ro = np.zeros(self.n_states)
        self.To = np.zeros((self.n_states, self.n_states))


    def save(self, path: str):
        
        pkl.dump(self.Q, open(os.path.join(f"{path}", "Q.pkl"), "wb"))
        pkl.dump(self.Ro, open(os.path.join(f"{path}", "Ro.pkl"), "wb"))
        pkl.dump(self.To, open(os.path.join(f"{path}", "To.pkl"), "wb"))

    @classmethod
    def load(cls, env, subgoal_index: int, gamma: float, path: str):
        """
        Instantiate an OptionBase (or subclass) and fill Q, Ro, To from disk.
        """
        # note: cls will be OptionVI or OptionQLearning
        opt = cls(env, subgoal_index, gamma)
        opt.Q  = pkl.load(open(os.path.join(path, "Q.pkl"),  "rb"))
        opt.Ro = pkl.load(open(os.path.join(path, "Ro.pkl"), "rb"))
        opt.To = pkl.load(open(os.path.join(path, "To.pkl"), "rb"))
        return opt

class OptionVI(OptionBase):

    def __init__(self, env, 
                 subgoal_index: int, 
                 gamma: Optional[float] = 1):
        
        super().__init__(env, subgoal_index, gamma)
        
    
    def train(self):
        
        Q = np.zeros((self.n_states, self.n_actions))
        V = np.zeros(self.n_states)

        To = np.zeros((self.n_states, self.n_states)) 

        To[:, self.subgoal_idx] = 1

        iters = 0

        while True:
            
            iters+=1

            Q_old = Q.copy()

            for state_idx in range(self.n_states):
                
                all_ns = []

                if state_idx == self.subgoal_idx:
                    continue

                for aidx in range(self.n_actions):

                    q_value = 0

                    for next_state_idx in range(self.n_states):
                       
                        prob = self.env.P[state_idx, aidx, next_state_idx]
                        if not prob:
                            continue
                        all_ns.append(next_state_idx)
                        done = next_state_idx == self.subgoal_idx

                        q_value += prob * (self.env.reward(self.env.state_to_coords[state_idx]) + (1-done) * self.gamma * V[next_state_idx]) 

                    Q[state_idx, aidx] = q_value
                
                To[state_idx, self.subgoal_idx] = self.gamma * np.max(To[all_ns, self.subgoal_idx])
            
            V = Q.max(axis=1)

            if np.allclose(Q_old, Q):
                break 
        
        self.Q = Q
        self.Ro = V
        self.To = To

        return iters
    
class OptionQLearning(OptionBase):

    def __init__(self, env, 
                 subgoal_index: int, 
                 gamma: Optional[float] = 1,
                 lr : float = 0.3,
                 init_epsilon: Optional[float] = 1,
                 final_epsilon: Optional[float] = 1,
                 warmup_steps: Optional[int] = 1000, 
                 learning_steps: Optional[int] = 2000):
        
        super().__init__(env, subgoal_index, gamma)
        
        self.lr = lr
        self.To = np.zeros((self.n_states, self.n_states))
        self.To[self.subgoal_idx, self.subgoal_idx] = 1
        self.init_epsilon = init_epsilon 
        self.final_epsilon = final_epsilon 
        self.warmup_steps = warmup_steps 
        self.learning_steps = learning_steps

        self.num_steps = 0

    
    def update_qfunction(self, 
                         state: tuple,
                         action: int,
                         next_state: tuple,
                         reward: float):

        if state != self.subgoal_state:

            done = next_state == self.subgoal_state

            state_idx = self.env.coords_to_state[state]
            next_state_idx =  self.env.coords_to_state[next_state]
            target = reward + self.gamma * (1 - done) * self.Q[next_state_idx, :].max()
            value = self.Q[state_idx, action]
            update = target - value
            self.Q[state_idx, action] += self.lr * update
            self.Ro[state_idx] = self.Q[state_idx, :].max()
            u1, u2 = self.To[state_idx, self.subgoal_idx], self.gamma * self.To[next_state_idx, self.subgoal_idx]
            self.To[state_idx, self.subgoal_idx] = np.max([u1, u2])

            return update, target, value, done


    def get_epsilon(self):

        '''
        Linearly decaying exploration rate *per option*!!
        '''

        steps_left = self.learning_steps + self.warmup_steps - self.num_steps
        bonus = (self.init_epsilon - self.final_epsilon) * steps_left / self.learning_steps
        bonus = np.clip(bonus, 0., 1. - self.final_epsilon)
        return self.final_epsilon + bonus



class MetaPolicy(ABC):


    def __init__(self, env, eval_env,
                fsa : FiniteStateAutomaton, 
                T: np.ndarray,
                gamma: Optional[float] = 1, 
                num_iters: Optional[float] = 100,
                eval_episodes: Optional[float] = 1 ):

        self.env = env
        self.eval_env  = eval_env
        self.fsa = fsa
        self.T = T
        self.gamma = gamma
        self.num_iters = num_iters
        self.eval_episodes = eval_episodes

        self.Q = None
        self.V = None

    @abstractmethod
    def learn_options(self):
        raise NotImplementedError 

    def save(self, path:str):
        
        for i, option in enumerate(self.options):

            fullpath = os.path.join(path, f"options/option{i}/")
            os.makedirs(fullpath)
            option.save(fullpath)
    
        pkl.dump(self.Q, open(os.path.join(path,"Q.pkl"), "wb"))
        pkl.dump(self.mu, open(os.path.join(path,"metapolicy.pkl"), "wb"))

    @classmethod
    def load(cls,
             env,
             eval_env,
             fsa: FiniteStateAutomaton,
             T: np.ndarray,
             base_dir: str,
             gamma: float,
             eval_episodes: int = 1,
             num_iters: int = 50):
        """
        Instantiate a MetaPolicy subclass and rehydrate its options, Q, and mu.
        """
        # 1) Make the empty container
        inst = cls(env, eval_env, fsa, T, gamma, num_iters, eval_episodes)

        # 2) Load each option in order
        options = []
        opts_dir = os.path.join(base_dir, "options")
        # assume folders option0, option1, …, optionN
        for name in sorted(os.listdir(opts_dir)):
            if not name.startswith("option"):
                continue
            idx = int(name.replace("option", ""))
            subpath = os.path.join(opts_dir, name)
            # the subgoal_index must match how you originally built them:
            subgoal_state = env.exit_states[idx]
            subgoal_idx = env.coords_to_state[subgoal_state]
            option = OptionBase.load(env, subgoal_idx, gamma, subpath)
            options.append(option)
        inst.options = options

        # 3) Load the meta‐Q and mu
        inst.Q  = pkl.load(open(os.path.join(base_dir, "Q.pkl"),         "rb"))
        inst.mu = pkl.load(open(os.path.join(base_dir, "metapolicy.pkl"), "rb"))
        return inst

    def train_metapolicy(self, 
                         record: bool = False,
                         iters : Optional[int] = None):

        if self.Q is None and self.V is None:
            self.Q = np.zeros((len(self.fsa.states), self.env.s_dim, len(self.options)))
            self.V = np.zeros((len(self.fsa.states), self.env.s_dim))

        # FSA rewards (1 everywhere except at terminal)
        R = np.ones(len(self.fsa.states))
        terminals = [self.fsa.is_terminal(s) for s in self.fsa.states]
        R[np.where(terminals)] = 0

        times = [0]

        num_iters = self.num_iters if iters is None else iters

        for j in range(num_iters):

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


            # Multiply by T before passing to next iteration (this masks the value function)
            self.V = np.sum(self.T * preV, axis=1)
            
            # For each iteration, evaluate the policy
            mu_aux = self.Q.argmax(axis=2)
            mu = {}

            elapsed_iter = time.time() - iter_start if record else None

            for (fsa_state_idx, state_idx) in np.ndindex(mu_aux.shape):
                f, s = self.fsa.states[fsa_state_idx], self.env.state_to_coords[state_idx]
                mu[(f, s)] = mu_aux[fsa_state_idx, state_idx]

            times.append(elapsed_iter)
            
            if record:
                
                successes, acc_rewards = [], []

                for _ in range(self.eval_episodes):
                    success, acc_reward = self.evaluate_metapolicy(reset=False)
                    successes.append(success)
                    acc_rewards.append(acc_reward)

                success = np.average(successes)
                acc_reward = np.average(acc_rewards)

                log_dict = {"evaluation/success": success, 
                            "evaluation/acc_reward": acc_reward,
                            "evaluation/iter": j,
                            "evaluation/time": np.sum(times),
                            }

                wb.log(log_dict)
        
        if record:

            success, acc_reward = self.evaluate_metapolicy(reset=False)

        mu_aux = self.Q.argmax(axis=2)
        mu = {}

        for (fsa_state_idx, state_idx) in np.ndindex(mu_aux.shape):
            f, s = self.fsa.states[fsa_state_idx], self.env.state_to_coords[state_idx]
            mu[(f, s)] = mu_aux[fsa_state_idx, state_idx]
        
        self.mu = mu
        
        return mu
    
    def evaluate_metapolicy(self, 
                            max_steps : Optional[int]        = 200,
                            max_steps_option : Optional[int] = 40,
                            log: Optional[bool]              = False,
                            reset: Optional[bool]            = True,
                            i: Optional[int] = None
                            ):

        if self.Q is None:
            self.train_metapolicy()

        acc_reward, success = 0, False
        num_steps = 0

        (f_state, state), p = self.eval_env.reset()

        options_used = 0

        while num_steps < max_steps:

            fsa_state_idx = self.fsa.states.index(f_state)
            state_idx = self.env.coords_to_state[state]

            qvalues = self.Q[fsa_state_idx, state_idx]    

            option = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

            options_used+=1
            
            first, steps_in_option, done = True, 0, False

            while (steps_in_option < max_steps_option and state != self.options[option].subgoal_state) or first:

                qvalues = self.options[option].Q[self.env.coords_to_state[state]]
                action = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

                (f_state, state), reward, done, info = self.eval_env.step(action)

                p = info["proposition"]

                num_steps+=1
                acc_reward += reward
                steps_in_option+=1

                if done or num_steps == max_steps:
                    break

                first = False

            if done:
                success = self.fsa.is_terminal(f_state)
                break
        
                    
        if reset:
            
            self.Q  = None
            self.V  = None 
            self.mu = None
        
        return success, acc_reward
    

# ANCHOR: metapolicy
class MetaPolicyVI(MetaPolicy):

    def __init__(self, env, eval_env, 
                 fsa: FiniteStateAutomaton, 
                 T: np.ndarray, 
                 gamma: Optional[float] = 1 , 
                 num_iters: Optional[int] = 50,
                 eval_episodes: Optional[int] = 1):
        
        super().__init__(env, eval_env, fsa, T, gamma, num_iters)

        self.num_iters = num_iters
        self.eval_env = eval_env

        self.options = self.learn_options()
        self.Q = None 
        self.V = None
        self.mu = None
    
    
    def learn_options(self):
        
        """
            Automatically learns an option (with ValueIteration) per exit state.
        """

        options = []

        for subgoal_state in self.env.exit_states.values():
            
            subgoal_idx = self.env.coords_to_state[subgoal_state] 
            option = OptionVI(self.env, subgoal_idx, self.gamma)
            num_iters = option.train()

            options.append(option)

        return options


class MetaPolicyQLearning(MetaPolicy):

    def __init__(self, env, eval_env, 
                 fsa: FiniteStateAutomaton,
                 T: np.ndarray, 
                 gamma: Optional[float] = 1., 
                 num_iters: Optional[int] = 50,
                 lr: Optional[float] = 0.2, 
                 num_episodes: Optional[int] = 200,
                 episode_length: Optional[int] = 100, 
                 eval_freq : Optional[int] = 50,
                 eval_episodes: Optional[int] = 50,
                 init_epsilon: Optional[float] = 1,
                 final_epsilon: Optional[float] = 1,
                 warmup_steps: Optional[int] = 1000, 
                 learning_steps: Optional[int] = 2000,
                 log:Optional[bool] = True):
        
        super().__init__(env, eval_env, fsa, T, gamma, num_iters)

        self.lr = lr

        self.options = []
        self.num_episodes = num_episodes 
        self.episode_length = episode_length
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        self.init_epsilon = init_epsilon,
        self.final_epsilon = final_epsilon,
        self.warmup_steps = warmup_steps, 
        self.learning_steps = learning_steps

        self.log = log

        exit_states_idxs = list(map(lambda x: self.env.coords_to_state[x], self.env.exit_states.values()))
        self.exit_states_idxs = exit_states_idxs

        self.define_wb_metrics()

        for option_idx, subgoal_idx in enumerate(self.exit_states_idxs):
            self.define_wb_metrics_option(option_idx)
            option = OptionQLearning(self.env, subgoal_idx, self.gamma, self.lr, init_epsilon, final_epsilon, warmup_steps, learning_steps)
            self.options.append(option)

    def get_epsilon_greedy_action(self, 
                                  option_idx : int, 
                                  state : tuple):
        
        epsilon = self.options[option_idx].get_epsilon()
        
        if np.random.rand() < epsilon:

            return np.random.randint(0, self.env.action_space.n)
        
        else:

            state_idx = self.env.coords_to_state[state]
            qvalues = self.options[option_idx].Q[state_idx, :]

            return np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

    def learn_options(self):
        
        num_options = len(self.options)

        total_steps = 0

        self.env.reset()

        for i in range(self.num_episodes):

            self.env.random_reset()

            # Use one option at a time
            option_idx = i % num_options   

            num_steps = 0

            while num_steps < self.episode_length:
                
                num_steps += 1
                total_steps += 1

                current_state  = self.env.state

                log_dict = {f"option_learning/option_{option_idx}/epsilon": self.options[option_idx].get_epsilon(), 
                             f"option_learning/option_{option_idx}/num_steps" : self.options[option_idx].num_steps}

                wb.log(log_dict)
                
                action = self.get_epsilon_greedy_action(option_idx, current_state)
                self.options[option_idx].num_steps += 1

                _, r, _, _ = self.env.step(action)
                next_state = self.env.state

                # Intra option learning
                for oidx, option in enumerate(self.options):
                    res = option.update_qfunction(current_state, action, next_state, r)

                if not self.eval_freq is None and total_steps % self.eval_freq == 0:
                    # Log the performance during training
                    success, reward = self.evaluate_metapolicy(i=num_steps)

                    log_dict = {
                        "learning/success" : int(success),
                        "learning/fsa_reward": reward,
                        "learning/timestep": total_steps,
                        "learning/episode": i, }

                    wb.log(log_dict)
                
                if total_steps % self.eval_freq == 0:

                    self.evaluate_options(total_steps)

        self.env.reset()

    
    def evaluate_options(self, 
                         num_step : int, 
                         max_steps : Optional[int]= 50):
        
        env = self.eval_env.env
        
        for i, option in enumerate(self.options):

            obs = tuple(env.reset())
            acc_reward = 0

            for _ in range(max_steps):

                obs = tuple(obs)

                qvalues = option.Q[env.coords_to_state[obs], :]
                action = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

                obs, reward, _, _ = env.step(action)

                obs = tuple(obs)
                
                acc_reward += reward

                if obs == option.subgoal_state:
                    break
            
            wb.log({f"option_learning/option_{i}/acc.reward": acc_reward, "learning/timestep" : num_step})

    def define_wb_metrics(self):
        wb.define_metric(f"learning/success", step_metric="learning/timestep")
        wb.define_metric(f"learning/fsa_reward", step_metric="learning/timestep")
        wb.define_metric(f"learning/episode", step_metric="learning/timestep")
        wb.define_metric(f"learning/timestep")


    def define_wb_metrics_option(sefl, option_idx):
        wb.define_metric(f"option_learning/option_{option_idx}/acc.reward", step_metric = f"option_learning/option_{option_idx}/num_steps")
        wb.define_metric(f"option_learning/option_{option_idx}/epsilon", step_metric = f"option_learning/option_{option_idx}/num_steps")
      
            
            
            
    
   