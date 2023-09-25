import gym 
import envs 
from algorithms.value_iteration import ValueIteration
from time import sleep

env = gym.make("DeliveryMini-v0")

obs = env.reset()
# obs = env.unwrapped.reset((7, 3))

algorithm = ValueIteration(env)

print(env.unwrapped.exit_states)

Q, V, To = algorithm.train(goal_state=(1, 2))


print(V)

for _ in range(100):
     state = tuple(obs)
     obs, reward, done, _ = env.step(Q[env.unwrapped.coords_to_state[state ]].argmax())
     print(obs, done)
     env.render()
     sleep(1)