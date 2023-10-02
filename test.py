import gym 
import envs 
from envs.wrapper import DeliveryAutomatonEnv
from time import sleep

from algorithms.options import MetapolicyVI
from task_specifications import *


env = gym.make("Delivery-v0")
eval_env = gym.make("DeliveryEval-v0")


fsa, T = fsa_delivery1(env)
eval_env = DeliveryAutomatonEnv(eval_env, fsa, "u0", T)

policy = MetapolicyVI(env, eval_env, fsa, T)
policy.train_metapolicy()


