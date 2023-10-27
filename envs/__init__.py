import gym

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
    id='OfficeComplex-v0',
    entry_point='envs.grid_envs:OfficeComplex',
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeComplexEval-v0',
    entry_point='envs.grid_envs:OfficeComplex',
    kwargs={'init_state': (6, 2)},
    max_episode_steps=200,
)
