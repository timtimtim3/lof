from stable_baselines3 import DQN
import gym
import torch as th
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union, List
from abc import ABC, abstractmethod
from typing import Optional
import pickle as pkl
import numpy as np
import wandb as wb
import time
import os

from fsa.fsa import FiniteStateAutomaton
from sfols.plotting.plotting import plot_q_vals
from sfols.plotting.plotting import plot_q_vals
from sfols.rl.rl_algorithm import RLAlgorithm
from sfols.rl.utils.buffer import ReplayBuffer
from sfols.rl.utils.prioritized_buffer import PrioritizedReplayBuffer
from sfols.rl.utils.utils import linearly_decaying_epsilon, polyak_update, huber
from sfols.rl.utils.nets import mlp


class SubgoalRewardEnv(gym.Wrapper):
    def __init__(self, env, subgoal_cells, reward_goal=True):
        """
        env            : your continuous GridEnvContinuous
        subgoal_cells : iterable of (row,col) tuples making up the goal area
        """
        super().__init__(env)
        self.env = env
        # Normalize to a set of discrete cells
        if not hasattr(subgoal_cells, "__iter__") or isinstance(subgoal_cells, tuple):
            subgoal_cells = [subgoal_cells]
        self.subgoal_cells = set(subgoal_cells)
        self.done          = False

        self.step_reward = 0 if reward_goal else -1
        self.goal_reward = 1 if reward_goal else -1

    def reset(self, **kwargs):
        self.done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        reward = self.step_reward

        if self.done:
            # once we've hit the goal, keep returning done
            return obs, reward, True, info

        # map continuous obs back to the discrete (row,col)
        state_cell = self.env.continuous_to_cell(obs)

        if state_cell in self.subgoal_cells:
            self.done = True
            return obs, self.goal_reward, True, info

        return obs, reward, False, info


class OptionDqnSB3:
    def __init__(self,
                 base_env: gym.Env,
                 subgoal_cells: set,
                 gamma: float        = 0.99,
                 total_timesteps: int = 100_000,
                 net_arch: list       = [128,128],
                 **dqn_kwargs):
        # Wrap to give +1 reward & terminate at the subgoal
        self.env = SubgoalRewardEnv(base_env, subgoal_cells)
        # Build the DQN
        self.model = DQN(
            "MlpPolicy",
            self.env,
            gamma=gamma,
            policy_kwargs=dict(net_arch=net_arch),
            **dqn_kwargs
        )
        self.total_timesteps = total_timesteps
        self.Q = None

    def train(self):
        # 1) learn the network
        self.model.learn(total_timesteps=self.total_timesteps)
        # 2) tabularize Q by querying the net at each cell-center
        n_states  = len(self.env.env.coords_to_state)
        n_actions = self.env.action_space.n
        Q_tab     = np.zeros((n_states, n_actions), dtype=np.float32)

        for cell, idx in self.env.env.coords_to_state.items():
            cont = self.env.env.cell_to_continuous_center(cell)
            obs  = th.as_tensor(cont, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.no_grad():
                qvals = self.model.q_net(obs).cpu().numpy().squeeze(0)
            Q_tab[idx] = qvals

        self.Q = Q_tab

    def act(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return int(a)

    def save(self, path: str):
        self.model.save(path)
        np.save(f"{path}/Q_table.npy", self.Q)

    @classmethod
    def load(cls, base_env, subgoal_cell, cell_size, path, **init_kwargs):
        opt = cls(base_env, subgoal_cell, cell_size, **init_kwargs)
        opt.model = DQN.load(path, env=opt.env)
        opt.Q     = np.load(f"{path}/Q_table.npy")
        return opt


class QNet(nn.Module):
    """
    Q-network with optional normalization of continuous inputs.
    Splits the input vector into:
      - cont: first `cont_dim` dims, normalized via (x - low)/ (high - low)
      - fsa: remaining `fsa_dim` dims (one-hot), left unchanged
    Then passes concatenated tensor through an MLP to output `action_dim` values.
    """

    def __init__(self,
                 cont_dim: int,
                 fsa_dim: int,
                 action_dim: int,
                 net_arch: List[int],
                 normalize_inputs: bool = False,
                 obs_low: np.ndarray = None,
                 obs_high: np.ndarray = None):
        super().__init__()
        self.cont_dim = cont_dim
        self.fsa_dim = fsa_dim
        self.action_dim = action_dim
        self.normalize_inputs = normalize_inputs
        input_dim = cont_dim + fsa_dim

        if self.normalize_inputs:
            assert obs_low is not None and obs_high is not None, \
                "Must provide obs_low and obs_high when normalize_inputs=True"
            self.register_buffer("obs_low", th.tensor(obs_low, dtype=th.float32))
            self.register_buffer("obs_high", th.tensor(obs_high, dtype=th.float32))

        self.net = mlp(input_dim, action_dim, net_arch)
        # optional initialization (if desired)
        # self.apply(layer_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [batch, cont_dim + fsa_dim]
        if self.normalize_inputs:
            cont = x[..., :self.cont_dim]
            denom = (self.obs_high - self.obs_low).clamp(min=1e-6)
            cont = (cont - self.obs_low) / denom
            fsa = x[..., self.cont_dim:]
            x = th.cat([cont, fsa], dim=-1)
        return self.net(x)  # [batch, action_dim]


class OptionDQN(RLAlgorithm):
    def __init__(self,
                 env,
                 subgoal_cells: set,
                 option_id: int,
                 meta,
                 net_arch: List[int] = [256, 256],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 initial_epsilon: float = 1.0,
                 final_epsilon: float = 0.1,
                 epsilon_decay_steps: Optional[int] = 10000,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 learning_starts: int = 1000,
                 target_update_freq: int = 1000,
                 tau: float = 1.0,
                 per: bool = False,
                 min_priority: float = 1.0,
                 normalize_inputs: bool = True,
                 log: bool = True,
                 log_prefix: str = "option_learning/",
                 device: Union[str, th.device] = 'auto',
                 eval_freq=500,
                 goal_prop=None,
                 **kwargs) -> None:
        super().__init__(env, device, fsa_env=None, log_prefix=log_prefix)
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.batch_size = batch_size
        self.per = per
        self.min_priority = min_priority
        self.Q = None
        self.option_id = option_id
        self.num_timesteps = 0
        self.log_prefix = log_prefix + f"option_{self.option_id}/"
        self.meta = meta
        self.eval_freq = eval_freq
        self.goal_prop = goal_prop

        self.n_states  = len(self.env.env.coords_to_state)
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.Ro = np.zeros(self.n_states)
        self.To = np.zeros((self.n_states, self.n_states))

        self.subgoal_indices = [env.coords_to_state[subgoal_cell] for subgoal_cell in subgoal_cells]
        self.subgoal_cells = subgoal_cells

        # initialize each goal cell to be absorbing
        for g in self.subgoal_indices:
            self.To[g, g] = 1.0

        # Wrap to give +1 reward & terminate at the subgoal
        self.env = SubgoalRewardEnv(env, subgoal_cells)

        # continuous coordinate bounds
        obs_low, obs_high = env.env.get_observation_bounds()
        # Build Q-net and target Q-net
        self.q_net = QNet(
            cont_dim=obs_low.shape[0],
            fsa_dim=0,
            action_dim=env.action_space.n,
            net_arch=net_arch,
            normalize_inputs=normalize_inputs,
            obs_low=obs_low,
            obs_high=obs_high
        ).to(self.device)
        self.target_q_net = QNet(
            cont_dim=obs_low.shape[0],
            fsa_dim=0,
            action_dim=env.action_space.n,
            net_arch=net_arch,
            normalize_inputs=normalize_inputs,
            obs_low=obs_low,
            obs_high=obs_high
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # replay buffer
        if per:
            self.replay_buffer = PrioritizedReplayBuffer(
                obs_dim=obs_low.shape[0] + 0,
                action_dim=1,
                rew_dim=1,
                max_size=buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                obs_dim=obs_low.shape[0] + 0,
                action_dim=1,
                rew_dim=1,
                max_size=buffer_size
            )

        # logging
        self.log = log
        if log:
            wb.define_metric(f"{log_prefix}epsilon", step_metric="learning/timestep")
            wb.define_metric(f"{log_prefix}critic_loss", step_metric="learning/timestep")
            wb.define_metric(f"eval/reward", step_metric="learning/timestep")

    def _build_input(self, obs: np.ndarray) -> th.Tensor:
        return th.tensor(obs, dtype=th.float32, device=self.device)

    def act(self, state: np.ndarray, deterministic=False) -> int:
        # random exploration
        if not deterministic and self.num_timesteps < self.learning_starts or np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        inp = self._build_input(state).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return int(q.argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        inp = self._build_input(state).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return q.cpu().numpy().squeeze(0)

    def sample_batch(self):
        return self.replay_buffer.sample(
            batch_size=self.batch_size,
            to_tensor=True,
            device=self.device
        )

    def learn(self,
              total_timesteps: int,
              total_episodes: Optional[int] = None,
              reset_num_timesteps: bool = False,
              **kwargs) -> None:
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_episodes = 0
            self.epsilon = self.initial_epsilon

        state = self.env.reset()
        done = False
        ep_ret = 0.0

        for t in range(1, total_timesteps + 1):
            self.num_timesteps += 1
            self.meta.total_steps += 1
            a = self.act(state)
            next_state, reward, done, _ = self.env.step(a)

            # store to buffer
            self.replay_buffer.add(state, a, reward, next_state, done)

            # update
            if self.num_timesteps >= self.learning_starts:
                obs_b, act_b, rew_b, nxt_b, done_b = self.sample_batch()
                q_vals = self.q_net(obs_b).gather(1, act_b.long()).squeeze(1)
                with th.no_grad():
                    next_q = self.target_q_net(nxt_b).max(1)[0]
                    target = rew_b + self.gamma * (1 - done_b) * next_q
                td = q_vals - target
                loss = huber(td)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                # target update
                if t % self.target_update_freq == 0 or self.tau < 1.0:
                    polyak_update(
                        self.q_net.parameters(),
                        self.target_q_net.parameters(),
                        self.tau
                    )

                self._update_ro(state)
                self._update_to(state, next_state)

                # epsilon decay
                if self.epsilon_decay_steps:
                    self.epsilon = linearly_decaying_epsilon(
                        self.initial_epsilon,
                        self.epsilon_decay_steps,
                        self.num_timesteps,
                        self.learning_starts,
                        self.final_epsilon
                    )
                # logging
                if self.log and t % self.eval_freq == 0:
                    self._update_tab_q()
                    success, reward = self.meta.evaluate_metapolicy()

                    # if total_steps % self.eval_freq == 0:
                    #     self.evaluate_options(total_steps)

                    wb.log({
                        "learning/success": int(success),
                        "learning/fsa_reward": reward,
                        "learning/total_timestep": self.meta.total_steps,
                        f"{self.log_prefix}epsilon": self.epsilon,
                        f"{self.log_prefix}critic_loss": loss.mean().item(),
                        f"{self.log_prefix}timestep": self.num_timesteps,
                    })

            # episode bookkeeping
            ep_ret += reward
            state = next_state
            if done:
                self.num_episodes += 1
                if self.log:
                    wb.log({
                        f"{self.log_prefix}episode_return": ep_ret,
                        f"{self.log_prefix}episode": self.num_episodes,
                        f"{self.log_prefix}timestep": self.num_timesteps
                    })
                state = self.env.reset()
                done = False
                ep_ret = 0.0

        self._update_tab_q()

    def _update_tab_q(self):
        # 2) tabularize Q by querying the net at each cell-center
        for cell, idx in self.env.env.coords_to_state.items():
            cont = self.env.env.cell_to_continuous_center(cell)
            obs  = th.as_tensor(cont, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.no_grad():
                qvals = self.q_net(obs).cpu().numpy().squeeze(0)
            self.Q[idx] = qvals

    def _update_ro(self, state):
        cell = self.env.env.continuous_to_cell(state)
        state_idx = self.env.env.coords_to_state[cell]

        cont = self.env.env.continuous_state_to_continuous_center(state)
        obs = th.as_tensor(cont, dtype=th.float32, device=self.device).unsqueeze(0)
        with th.no_grad():
            qvals = self.q_net(obs).cpu().numpy().squeeze(0)

        # Ro still just max‐over‐actions of the new Q
        self.Ro[state_idx] = qvals.max()

    def _update_to(self, state, next_state):
        cell = self.env.env.continuous_to_cell(state)
        state_idx = self.env.env.coords_to_state[cell]

        next_cell = self.env.env.continuous_to_cell(next_state)
        next_state_idx = self.env.env.coords_to_state[next_cell]

        # now update *all* the termination‐columns for the group
        for g in self.subgoal_indices:
            # u1 = old To[s,g], u2 = γ·To[s_next, g]
            u1 = self.To[state_idx, g]
            u2 = self.gamma * self.To[next_state_idx, g]
            self.To[state_idx, g] = max(u1, u2)

    def get_epsilon(self):
        epsilon = linearly_decaying_epsilon(
            self.initial_epsilon,
            self.epsilon_decay_steps,
            self.num_timesteps,
            self.learning_starts,
            self.final_epsilon
        )
        return epsilon

    def train(self, *args, **kwargs):
        """
        Stub to satisfy the abstract RLAlgorithm interface.
        All of our Q-learning actually runs inside `learn()`, so we don’t
        need this, but we must define it.
        """
        return None

    def get_config(self) -> dict:
        """
        Return a serializable dict of hyperparameters for logging or checkpointing.
        """
        return {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_update_freq": self.target_update_freq,
            "tau": self.tau,
            "per": self.per,
            "net_arch": self.q_net.net.hidden_layers if hasattr(self.q_net, "net") else None,
            "normalize_inputs": self.q_net.normalize_inputs,
        }

    def eval(self, state: np.ndarray) -> int:
        inp = self._build_input(state).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return int(q.argmax(dim=1).item())

    def save(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"dqn_option{self.option_id}.pt")
        th.save({
            'q_state': self.q_net.state_dict(),
            'target_q_state': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str, option_id: int):
        # pick device
        device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

        # load onto correct device
        data = th.load(os.path.join(path, f"dqn_option{option_id}.pt"), map_location=device)

        # restore and move nets
        self.q_net.load_state_dict(data['q_state'])
        self.q_net.to(device)

        self.target_q_net.load_state_dict(data['target_q_state'])
        self.target_q_net.to(device)

        # optimizer
        self.optimizer.load_state_dict(data['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.to(device)

    def get_arrow_data(self, batch_size: int = 256):
        """
        Returns the quiver‐plot params (X,Y,U,V,C) for the greedy policy
        in FSA‐state `uidx` over all continuous centers.
        """
        # 1) collect all cell‐centers
        centers = self.env.env.get_all_valid_continuous_states_centers()  # list of (y,x)
        N = len(centers)

        # 2) build a tensor input for each center
        #    _build_input returns a torch.Tensor of shape [cont_dim+fsa_dim]
        inputs = [ self._build_input(np.array(center))
                   for center in centers ]

        all_actions = []
        all_qvals   = []

        # 3) batch through q_net
        for start in range(0, N, batch_size):
            batch = inputs[start : start + batch_size]  # list of Tensors
            batch_tensor = th.stack(batch, dim=0)        # [B, input_dim]
            with th.no_grad():
                q_out = self.q_net(batch_tensor)        # [B, action_dim]
            # pick greedy actions & values
            acts = q_out.argmax(dim=1).cpu().numpy()   # shape [B,]
            qmax = q_out.max(dim=1).values.cpu().numpy()
            all_actions.append(acts)
            all_qvals.append(qmax)

        # 4) concatenate back to full length
        actions = np.concatenate(all_actions, axis=0)  # [N,]
        qvals   = np.concatenate(all_qvals,   axis=0)  # [N,]

        # 5) hand off to your env’s quiver‐builder
        return self.env.env.get_arrow_data(actions, qvals, states=centers)

    def plot_q_vals(self, activation_data=None, base_dir=None, show=True):
        def _plot_one():
            save_path = f"{base_dir}/qvals_option{self.option_id}.png" if base_dir is not None else None
            arrow_data = self.get_arrow_data()
            plot_q_vals(self.env.env, arrow_data=arrow_data, activation_data=activation_data,
                        save_path=save_path, show=show, goal_prop=self.goal_prop)

        _plot_one()


class MetaPolicyContinuous(ABC):
    def __init__(self, env, eval_env,
                 fsa: FiniteStateAutomaton,
                 T: np.ndarray,
                 gamma: Optional[float] = 1,
                 num_iters: Optional[float] = 100,
                 eval_episodes: Optional[float] = 1):

        self.env = env
        self.eval_env = eval_env
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

    def save(self, path: str):

        for option in self.options:
            fullpath = os.path.join(path, f"options/option{option.option_id}/")
            os.makedirs(fullpath)
            option.save(fullpath)

        pkl.dump(self.Q, open(os.path.join(path, "Q.pkl"), "wb"))
        pkl.dump(self.mu, open(os.path.join(path, "metapolicy.pkl"), "wb"))

    def train_metapolicy(self,
                         record: bool = False,
                         iters: Optional[int] = None):

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
                            max_steps: Optional[int] = 200,
                            max_steps_option: Optional[int] = 40,
                            log: Optional[bool] = False,
                            reset: Optional[bool] = True,
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
            state_cell = self.env.continuous_to_cell(state)
            state_idx = self.env.coords_to_state[state_cell]

            qvalues = self.Q[fsa_state_idx, state_idx]

            option = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

            options_used += 1

            first, steps_in_option, done = True, 0, False

            while (steps_in_option < max_steps_option and tuple(state) not in self.options[option].subgoal_cells) or first:

                state_cell = self.env.continuous_to_cell(state)
                state_idx = self.env.coords_to_state[state_cell]
                qvalues = self.options[option].Q[state_idx]
                action = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

                (f_state, state), reward, done, info = self.eval_env.step(action)

                p = info["proposition"]

                num_steps += 1
                acc_reward += reward
                steps_in_option += 1

                if done or num_steps == max_steps:
                    break

                first = False

            if done:
                success = self.fsa.is_terminal(f_state)
                break

        if reset:
            self.Q = None
            self.V = None
            self.mu = None

        return success, acc_reward

    def load(self, base_dir: str):
        """
        Instantiate a MetaPolicy subclass and rehydrate its options, Q, and mu.
        """
        opts_dir = os.path.join(base_dir, "options")
        # assume folders option0, option1, …, optionN
        for name in sorted(os.listdir(opts_dir)):
            if not name.startswith("option") or ".png" in name:
                continue
            idx = int(name.replace("option", ""))
            subpath = os.path.join(opts_dir, name)

            option = self.options[idx]
            option.load(path=subpath, option_id=idx)

        # 3) Load the meta‐Q and mu
        self.Q = pkl.load(open(os.path.join(base_dir, "Q.pkl"), "rb"))
        self.mu = pkl.load(open(os.path.join(base_dir, "metapolicy.pkl"), "rb"))

    def plot_q_vals(self, activation_data=None, base_dir: Optional[str] = None, show: bool = True,
                    policy_id: Optional[int] = None):
        """
        Plot the greedy-policy Q-value arrows for one or all options.

        Args:
            activation_data: extra overlay data passed to the plot util.
            base_dir:        if provided, each plot is saved to
                             "{base_dir}/option_{i}_qvals.png"
            show:            whether to display the figure.
            policy_id:       index of a single option to plot; if None, plots all.
        """
        # Helper to plot one option
        def _plot_one(opt):
            # delegates to OptionBase.plot_q_vals
            opt.plot_q_vals(
                activation_data=activation_data,
                base_dir=base_dir,
                show=show
            )

        if policy_id is not None:
            _plot_one(self.options[policy_id])
        else:
            for opt in self.options:
                _plot_one(opt)

    def plot_meta_qvals(self, activation_data=None, base_dir=None):
        states = self.env.get_planning_states()

        for uidx in range(len(self.fsa.states) - 1):
            actions, qvals, option_indices = [], [], []
            for state in states:
                state_cell = self.env.continuous_to_cell(state)
                state_idx = self.env.coords_to_state[state_cell]

                # 1) pick best option under meta-Q
                meta_q = self.Q[uidx, state_idx, :]
                opt_idx = int(np.random.choice(np.argwhere(meta_q == np.max(meta_q)).flatten()))
                option_indices.append(opt_idx)

                # 2) within that option, pick its greedy primitive action & Q-value
                prim_qs = self.options[opt_idx].Q[state_idx, :]
                prim_act = int(np.argmax(prim_qs))
                actions.append(prim_act)

                prim_q = float(prim_qs[prim_act])
                qvals.append(prim_q)

            save_path = f"{base_dir}/meta_u{uidx}.png" if base_dir is not None else None
            arrow_data = self.env.get_arrow_data(np.array(actions), np.array(qvals), states)
            plot_q_vals(self.env, arrow_data=arrow_data, activation_data=activation_data,
                        policy_indices=option_indices, save_path=save_path)


class MetaPolicyDQN(MetaPolicyContinuous):
    def __init__(self, env, eval_env,
                 fsa: FiniteStateAutomaton,
                 T: np.ndarray,
                 gamma: Optional[float] = 1.,
                 num_iters: Optional[int] = 50,
                 lr: Optional[float] = 0.2,
                 num_episodes: Optional[int] = 200,
                 epsilon_decay_steps=1000,
                 episode_length: Optional[int] = 100,
                 eval_freq: Optional[int] = 50,
                 eval_episodes: Optional[int] = 50,
                 init_epsilon: float = 1,
                 final_epsilon: float = 1,
                 warmup_steps: int = 1000,
                 learning_steps: Optional[int] = 2000,
                 log: Optional[bool] = True,
                 normalize_inputs=True,
                 per=False,
                 net_arch=[256, 256]
                 ):

        super().__init__(env, eval_env, fsa, T, gamma, num_iters)

        self.lr = lr

        self.options = []
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.epsilon_decay_steps = epsilon_decay_steps
        self.per = per
        self.normalize_inputs = normalize_inputs
        self.net_arch = net_arch

        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.warmup_steps = warmup_steps
        self.learning_steps = learning_steps
        self.total_steps = 0

        self.log = log

        self.define_wb_metrics()

        for prop_idx, subgoal_cells in sorted(env.exit_states.items()):
            self.define_wb_metrics_option(prop_idx)
            option = OptionDQN(self.env, subgoal_cells, option_id=prop_idx, meta=self, learning_rate=self.lr, gamma=self.gamma,
                               init_epsilon=self.init_epsilon, final_epsilon=self.final_epsilon,
                               epsilon_decay_steps=self.epsilon_decay_steps, learning_starts=self.warmup_steps,
                               per=self.per, normalize_inputs=self.normalize_inputs, net_arch=self.net_arch, 
                               goal_prop=env.PHI_OBJ_TYPES[prop_idx])
            # option = OptionQLearning(self.env, subgoal_idx, self.gamma, self.lr, init_epsilon, final_epsilon,
            #                          warmup_steps, learning_steps)
            self.options.append(option)

    def get_epsilon_greedy_action(self,
                                  option_idx: int,
                                  state: tuple):
        epsilon = self.options[option_idx].get_epsilon()

        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            qvalues = self.options[option_idx].q_values(state)
            return np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

    def learn_options(self):
        self.total_steps = 0
        for oidx, option in enumerate(self.options):
            self.options[oidx].learn(total_timesteps=self.learning_steps)
        self.env.reset()

    def evaluate_options(self,
                         num_step: int,
                         max_steps: Optional[int] = 50):

        env = self.eval_env.env

        for i, option in enumerate(self.options):

            obs = tuple(env.reset())
            acc_reward = 0

            for _ in range(max_steps):

                obs = tuple(obs)

                qvalues = option.q_values(np.array(obs))
                action = np.random.choice(np.argwhere(qvalues == np.amax(qvalues)).flatten())

                obs, reward, _, _ = env.step(action)

                obs = tuple(obs)

                acc_reward += reward

                if obs == option.subgoal_state:
                    break

            wb.log({f"option_learning/option_{i}/acc.reward": acc_reward, "learning/timestep": num_step})

    def define_wb_metrics(self):
        wb.define_metric(f"learning/success", step_metric="learning/timestep")
        wb.define_metric(f"learning/fsa_reward", step_metric="learning/timestep")
        wb.define_metric(f"learning/episode", step_metric="learning/timestep")
        wb.define_metric(f"learning/timestep")

    def define_wb_metrics_option(self, option_idx):
        wb.define_metric(f"option_learning/option_{option_idx}/acc.reward",
                         step_metric=f"option_learning/option_{option_idx}/num_steps")
        wb.define_metric(f"option_learning/option_{option_idx}/epsilon",
                         step_metric=f"option_learning/option_{option_idx}/num_steps")
