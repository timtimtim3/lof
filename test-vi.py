import gym 
import envs 
from envs.wrapper import DeliveryAutomatonEnv
from time import sleep

from algorithms.options import MetaPolicyVI
from torch.utils.tensorboard import SummaryWriter
from task_spec import load_fsa
import wandb
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--fsa")

args = parser.parse_args()
fsa_name = args.fsa

# TODO: Load experiment configs
project_name = "Logical-Options-Framework"
config = {"alpha":0.2,
          "epsilon": 0.3, 
          "num_episodes_training": 400,
          "episode_length": 100,
          "eval_freq": 500,
          "env": "Delivery-v0",
          "eval_env": "DeliveryEval-v0",
          "fsa": fsa_name,}

experiment_name = "lof-qlearning-vi-" + fsa_name

wandb.init(project=project_name,
                sync_tensorboard=True,
                config=config,
                name=experiment_name,
                monitor_gym=True,
                save_code=True)

writer = SummaryWriter(f"/tmp/{experiment_name}")

alpha = config["alpha"]
epsilon = config["epsilon"]
num_episodes = config["num_episodes_training"]
episode_length = config["episode_length"]
eval_freq = config["eval_freq"]
env = config["env"]
eval_env = config["eval_env"]
fsa_name = config["fsa"]

env = gym.make(env)
eval_env = gym.make(eval_env)
fsa, T = load_fsa(fsa_name, env)
eval_env = DeliveryAutomatonEnv(eval_env, fsa,  "u0", T)

policy = MetaPolicyVI(env, eval_env, fsa, T, gamma=1., record=True, num_iters = 50, writer= writer)

policy._learn_options()

policy.train_metapolicy(record=True)


writer.close()
wandb.finish()