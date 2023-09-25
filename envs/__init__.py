import gym

gym.envs.register(
    id='DeliveryMini-v0',
    entry_point='envs.grid_envs:DeliveryMini',
    max_episode_steps=200,
)
