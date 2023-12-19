import hydra 
import wandb 
from utils import seed_everything 
import gym 
import envs
from task_spec import load_fsa
import os, argparse
import pickle as pkl
import itertools
import pandas as pd

from algorithms.options import MetaPolicyVI, OptionBase

import matplotlib.pyplot as plt
import numpy as np

def read_file(filepath):
    
    with open(filepath, "rb") as file:
        content = pkl.load(file)
        return content


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--fsa_prefix", type=str)
    parser.add_argument("--seed", type=int, default=42)

    MAX_ITERS = 50


    args = parser.parse_args()
    run_name = args.run_name
    fsa_prefix = args.fsa_prefix
    seed = args.seed

    seed_everything(seed)

    # Init wandb
    newconfig = {
        "options_run_name": run_name,
    }

    # run = wandb.init(
    #     config=newconfig,
    #     entity="davidguillermo", 
    #     project="sfcomp",
    #     tags=["lof-multiple"]
    # )

    api = wandb.Api()

    # Retrieve run details from cloud and load SFs
    runs =  api.runs(path="davidguillermo/sfcomp")
    olderrun = list(filter(lambda run: str(run.name) == run_name, runs))[0]
    config = olderrun.config

    env_cfg = config.pop("env")


    env_name = env_cfg.pop("gym_name")
    eval_name = env_cfg.pop("eval_name")
    env = gym.make(env_name)
    eval_env = gym.make(eval_name)
    eval_env_target = config=env_cfg.pop("eval_env")

    # Retrieve pre-computed options

    all_p_reward, all_p_success = [], []
   
    tasks = [f"{fsa_prefix}{i}" for i in range(1, 4)]
       
    for t in tasks:

        res_acc_reward, res_success = [], []
    
        fsa, T = load_fsa(t, env)
        fsa_env = hydra.utils.call(eval_env_target, env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)

        mp = MetaPolicyVI(env, fsa_env, fsa, T)
        print(t, "options learned.")

        for i in range(50):
            print(t, i)
            mp.train_metapolicy(iters = 1)
            success, acc_reward = mp.evaluate_metapolicy(reset=False)
            res_acc_reward.append(acc_reward)
            res_success.append(success)
        
        all_p_reward.append(res_acc_reward)
        all_p_success.append(all_p_success)

    #     run.log({ 'metrics/evaluation/acc_reward': acc_reward,
    #               'metrics/evaluation/success': success,
    #               'metrics/evaluation/iter': i})
    # wandb.finish()


    rewards = np.vstack(all_p_reward)

    df = pd.DataFrame({'iter': list(range(50)),
                        'mean': rewards.mean(axis=0),
                        'std': rewards.std(axis=0)})

    df.to_csv("LOF-Office-redapt.csv")

if __name__ == "__main__":
    main()

