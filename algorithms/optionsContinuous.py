import numpy as np
from stable_baselines3 import DQN
import torch
import gym


class SubgoalRewardEnv(gym.Wrapper):
    def __init__(self, env, subgoal_cells):
        """
        env            : your continuous GridEnvContinuous
        subgoal_cells : iterable of (row,col) tuples making up the goal area
        """
        super().__init__(env)
        # Normalize to a set of discrete cells
        if not hasattr(subgoal_cells, "__iter__") or isinstance(subgoal_cells, tuple):
            subgoal_cells = [subgoal_cells]
        self.subgoal_cells = set(subgoal_cells)
        self.done          = False

    def reset(self, **kwargs):
        self.done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, _, info = self.env.step(action)

        if self.done:
            # once we've hit the goal, keep returning done
            return obs, 0.0, True, info

        # map continuous obs back to the discrete (row,col)
        state_cell = self.env.continuous_to_cell(obs)

        if state_cell in self.subgoal_cells:
            self.done = True
            return obs, 1.0, True, info

        return obs, 0.0, False, info


class OptionDQN:
    def __init__(self,
                 base_env: gym.Env,
                 subgoal_cell: tuple,
                 cell_size: float,
                 gamma: float        = 0.99,
                 total_timesteps: int = 100_000,
                 net_arch: list       = [128,128],
                 **dqn_kwargs):
        # Wrap to give +1 reward & terminate at the subgoal
        self.env = SubgoalRewardEnv(base_env, subgoal_cell, cell_size)
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
            obs  = torch.as_tensor(cont, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
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