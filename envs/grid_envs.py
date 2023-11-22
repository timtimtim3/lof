import numpy as np
import random
import gym
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod


class GridEnv(ABC, gym.Env):
    
    metadata = {'render.modes': ['human'],
        'video.frames_per_second': 20}
    
    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    @property
    def MAP(self):
        raise NotImplementedError

    @property
    def PHI_OBJ_TYPES(self):
        raise NotImplementedError
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    References
    ----------
    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start, random_act_prob, init_state=None):
        """
        Creates a new instance of the coffee environment.

        """
        self.random_act_prob = random_act_prob
        self.add_obj_to_start = add_obj_to_start

        self.viewer = None
        self.height, self.width = self.MAP.shape
        self.all_objects = dict(zip(self.PHI_OBJ_TYPES, range(len(self.PHI_OBJ_TYPES))))
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()

        self.init_state = init_state

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == '_':
                    self.initial.append((r, c))
                elif self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                elif self.MAP[r, c] in self.PHI_OBJ_TYPES:
                    self.object_ids[(r, c)] = len(self.object_ids)
                    if add_obj_to_start:
                        self.initial.append((r, c))

        self.exit_states = len(self.object_ids) * [None]
        for s in self.object_ids:
            symbol = self.MAP[s]
            self.exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        self.w = np.zeros(self.feat_dim)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.zeros(
            2), high=np.ones(2))
        self.seed()

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        """
        self.states = []
        idx = 0
        for i in range(0, self.MAP.shape[0]):
            for j in range(0, self.MAP.shape[1]):
                if self.MAP[i][j] == "X":
                    continue
                self.states.append((i, j))

    @abstractmethod
    def _create_transition_function(self):
        raise NotImplementedError

    def _create_transition_function_base(self):
        # Basic movement
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))
        for start_s in range(self.s_dim):
            for eff_a in range(self.a_dim):
                start_coords = self.states[start_s]
                new_coords = self.base_movement(start_coords, eff_a)
                new_s = self.states.index(new_coords)
                for a in range(self.a_dim):
                    if eff_a == a:
                        self.P[start_s, a, new_s] += 1-self.random_act_prob
                    else:
                        self.P[start_s, a, new_s] += self.random_act_prob / (self.a_dim-1)
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)

    @staticmethod
    def state_to_array(state):
        return np.array(state, dtype=np.int32)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(2147483647)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        if self.init_state is not None:
            self.state = self.init_state
        else:
            self.state = random.choice(self.initial)
        return self.state_to_array(self.state)
    
    def random_reset(self):
        states = [s for s in self.states if s not in self.exit_states]

        random_idx = np.random.randint(0, len(states))

        self.state = states[random_idx] 

        return self.state_to_array(self.state)

    def base_movement(self, coords, action):
        row, col = coords

        if action == self.LEFT:
            col -= 1
        elif action == self.UP:
            row -= 1
        elif action == self.RIGHT:
            col += 1
        elif action == self.DOWN:
            row += 1
        else:
            raise Exception('bad action {}'.format(action))
        if col < 0 or col >= self.width or row < 0 or row >= self.height or (row, col) in self.occupied: # no move
            return coords
        else:
            return (row, col)

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.states.index(old_state)
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.states[new_state_index]

        self.state = new_state

        # Determine features and rewards
        phi = self.features(old_state, action, new_state)
        reward = np.dot(phi, self.w)
        done = self.is_done(old_state, action, new_state)
        return self.state_to_array(self.state), reward, done, {'phi': phi, 'proposition': self.MAP[new_state]}

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        # (y, x), coll = state
        # n_state = self.width + self.height
        # result = np.zeros((n_state + len(coll),))
        # result[y] = 1
        # result[self.height + x] = 1
        # result[n_state:] = np.array(coll)
        # result = result.reshape((1, -1))
        # return result
        raise NotImplementedError()

    def encode_dim(self):
        return self.width + self.height + len(self.object_ids)

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def is_done(self, state, action, next_state):
        return next_state in self.object_ids

    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        return phi

    @property
    def feat_dim(self):
        return len(self.all_objects)

    @property
    def a_dim(self):
        return self.action_space.n

    @property
    def s_dim(self):
        return len(self.states)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.MAP.shape
            square_size = 55

            screen_height = square_size * max_x
            screen_width = square_size * max_y
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.square_map = {}
            for i in range(max_x):
                for j in range(max_y):
                    l = j * square_size
                    r = l + square_size
                    t = max_x * square_size - i * square_size
                    b = t - square_size
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(square)
                    self.viewer.square_map[(i, j)] = square

        for square_coords in self.viewer.square_map:
            square = self.viewer.square_map[square_coords]

            # Agent
            if square_coords == tuple(self.state):
                color = [1, 1, 0]

            # Exit state
            elif square_coords in self.object_ids.keys():
                color = [0, 0, 1]
            elif square_coords in self.occupied:
                color = [0, 0, 0]
            else:
                color = [1, 1, 1]
            square.set_color(*color)
        self.custom_render(square_map=self.viewer.square_map)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def custom_render(self, square_map: dict[tuple[int, int]]):
        pass

    @abstractmethod
    def reward(self, state):
        raise NotImplementedError

    def _make_rewards(self):

        rewards = np.zeros(self.s_dim)

        for i, s in enumerate(self.states):
            rewards[i] = self.reward(s)

        return rewards


class Delivery(GridEnv):
    
    MAP = np.array([['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', 'C', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    [' ', 'A', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ','  ', 'B', ' ', ' ', ' ' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],
                    ['O', 'O', 'O', ' ', 'O', 'O', 'O', 'H', 'O', 'O', 'O', ' ', 'O', 'O', 'O' ],])
    
    PHI_OBJ_TYPES = ['A', 'B', 'C', 'H']
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start=True, random_act_prob=0.0, init_state=None):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob, init_state=init_state)
        self._create_coord_mapping()
        self._create_transition_function()

        self.obstacles = []

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'X':
                    self.obstacles.append((r, c))

        self.rewards = self._make_rewards()

    def _create_transition_function(self):
        self._create_transition_function_base()

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.states.index(old_state)
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.states[new_state_index]

        self.state = new_state

        # Determine features and rewards
        reward = self.reward(old_state)

        done = False

        prop = self.MAP[new_state]
        
        return self.state_to_array(self.state), reward, done, {'proposition' : prop}

    def reward(self, state):

        reward = -1 

        y, x = state

        if self.MAP[y][x] == 'O':
            reward = -1000 

        return reward


    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        elif self.MAP[s1] == 'O':
            phi[:] = -100
        
        return phi

    
    def custom_render(self, square_map: dict[tuple[int, int]]):
        for square_coords in square_map:
            square = square_map[square_coords]
            # Teleport
            if self.MAP[square_coords] == 'O' :
                color = [0, 0, 0]
            elif self.MAP[square_coords] == 'A' :
                color = [1, 0, 0]
            elif self.MAP[square_coords] == 'B' :
                color = [0, 1, 0]
            elif self.MAP[square_coords] == 'H' :
                color = [0.7, 0.3, 0.7]
            else:
                continue
            square.set_color(*color)


class OfficeComplex(GridEnv):
    MAP = np.array([[' ', ' ', ' ',   'X', ' ', 'C2', ' ', ' '],
                     [' ', ' ', 'C1', 'X', 'X', ' ', ' ', ' '],
                     ['M2', ' ', ' ',  ' ', 'X', 'O2', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  'X', ' ', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  'X', ' ', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  ' ', ' ', ' ', ' '],
                     [' ', ' ', '_',  ' ',  ' ', ' ', ' ', ' '],
                     ['O1', ' ',' ',  ' ',  ' ', ' ', ' ', 'M1'], ])
    
    PHI_OBJ_TYPES = ['C1', 'C2', 'O1', 'O2', 'M1', 'M2']
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start=True, random_act_prob=0.0, init_state=None):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob, init_state=init_state)
        self._create_coord_mapping()
        self._create_transition_function()

        self.obstacles = []

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'X':
                    self.obstacles.append((r, c))

        self.exit_states = len(self.object_ids) * [None]
        for s in self.object_ids:
            symbol = self.MAP[s]
            self.exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        self.rewards = self._make_rewards()


    def _make_rewards(self):

        rewards = np.zeros(self.s_dim)

        for i, s in enumerate(self.states):
            rewards[i] = self.reward(s)

        return rewards
    

    def _create_transition_function(self):
        self._create_transition_function_base()

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.states.index(old_state)
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.states[new_state_index]

        self.state = new_state

        # Determine features and rewards
        reward = self.reward(old_state)

        done = False

        prop = self.MAP[new_state]
        
        return self.state_to_array(self.state), reward, done, {'proposition' : prop}

    def reward(self, state):

        reward = -1 

        y, x = state

        if self.MAP[y][x] == 'X':
            reward = -1000 

        return reward


    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        elif self.MAP[s1] == 'X':
            phi[:] = -100
        
        return phi


class DoubleSlit(GridEnv):
    MAP = np.array([
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O1', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['_', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O2', 'X']

    ])
    PHI_OBJ_TYPES = ['O1', 'O2']
    UP, RIGHT, DOWN = 0, 1, 2

    def __init__(self, random_act_prob=0.0, add_obj_to_start=False, init_state=None, max_wind=1):
        """
        Creates a new instance of the coffee environment.

        """
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob, init_state=init_state)
        self.action_space = Discrete(3)
        self._max_wind = max_wind
        self._create_coord_mapping()
        self._create_transition_function()
        self.rewards = self._make_rewards()
        

    def coords_act_transition_distr(self, coords, action):
        row, col = coords
        distr = []
        for wind in range(-self._max_wind, self._max_wind + 1, 1):
            new_row = row
            new_col = col

            vert_move = wind + (action == self.UP) - (action == self.DOWN)
            horiz_move = 1 + (action == self.RIGHT)

            # Check vert move
            direction = -1 if vert_move < 0 else 1
            while vert_move != 0:
                vert_move -= direction
                if (new_row + direction, new_col) not in self.occupied:
                    new_row = min(self.height - 1, new_row + direction)
                    new_row = max(0, new_row)

            # Check horiz move
            direction = -1 if horiz_move < 0 else 1
            while horiz_move != 0:
                horiz_move -= direction
                if (new_row, new_col + direction) not in self.occupied:
                    new_col = min(self.width - 1, new_col + direction)
                    new_col = max(0, new_col)

            entry = ((new_row, new_col), 1.0/(self._max_wind * 2 + 1))
            distr.append(entry)
        return distr

    def _create_transition_function(self):
        # Basic movement
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))
        for start_s in range(self.s_dim):
            for eff_a in range(self.a_dim):
                start_coords = self.states[start_s]
                # if start_coords in self.object_ids:
                #     self.P[start_s, eff_a, start_s] += 1  # Set transitions in goal states to 1 to pass the sanity check
                #     continue
                distr = self.coords_act_transition_distr(coords=start_coords, action=eff_a)
                for end_coords, prob in distr:
                    new_s = self.states.index(end_coords)
                    self.P[start_s, eff_a, new_s] += prob
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)


    def reward(self, state):

        reward = -1

        y, x = state

        if self.MAP[y][x] == 'O':
            reward = -1000

        return reward
