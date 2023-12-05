import gym

gym.envs.register(
    id='DoubleSlit-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlit-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
     kwargs={'max_wind': 1},
)

gym.envs.register(
    id='DoubleSlitEval-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={'init_state': (10, 0), 'max_wind': 1},
)

gym.envs.register(
    id='DoubleSlitEval-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={'init_state': (10, 0), 
            'max_wind': 3},
)


gym.envs.register(
    id='DeliveryMini-v0',
    entry_point='envs.grid_envs:DeliveryMini',
    max_episode_steps=200,
)

gym.envs.register(
    id='Delivery-v0',
    entry_point='envs.grid_envs:Delivery',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryEval-v0',
    entry_point='envs.grid_envs:Delivery',
    kwargs={'init_state': (14, 7)},
    max_episode_steps=200,
)

gym.envs.register(
    id='Office-v0',
    entry_point='envs.grid_envs:Office',
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeEval-v0',
    entry_point='envs.grid_envs:Office',
    kwargs={'init_state': (9, 3)},
    max_episode_steps=200,
)
