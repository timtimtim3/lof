import hydra 
from omegaconf import DictConfig, OmegaConf
import wandb 
from utils import seed_everything 
import gym 
import envs
from task_spec import load_fsa
from torch.utils.tensorboard import SummaryWriter
import os, argparse
import pickle as pkl

from algorithms.options import MetaPolicyQLearning, OptionBase


def read_file(filepath):
    
    with open(filepath, "rb") as file:
        content = pkl.load(file)
        return content


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--task", type=str, default="task1")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    num_iters = args.num_iters
    task = args.task
    run_name = args.run_name
    seed = args.seed

    seed_everything(seed)


    # Init wandb
    newconfig = {
        "num_iters": num_iters,
        "options_run_name": run_name,
    }

    run = wandb.init(
        config=newconfig,
        entity="davidguillermo", 
        project="sfcomp",
        tags=["lof-fsa"]
    )

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
    fsa, T = load_fsa(config.pop("fsa_name"), env)

    eval_env = hydra.utils.call(config=env_cfg.pop("eval_env"), env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)   

    # Retrieve pre-computed options
    dir = os.path.join("results", run_name)

    metapolicy = os.path.join(dir, "metapolicy.pkl")
    Q =  os.path.join(dir, "Q.pkl")

    options = []
    
    options_dir = os.path.join("results", run_name, "options")

    for (option, subgoal) in zip(os.listdir(options_dir), env.exit_states):
        o = OptionBase(env, subgoal, gamma=1)
        o.Q = read_file(os.path.join(options_dir, option, "Q.pkl"))
        o.Ro = read_file(os.path.join(options_dir, option, "Ro.pkl"))
        o.To = read_file(os.path.join(options_dir, option, "To.pkl"))
        options.append(o)

    mp = MetaPolicyQLearning(env, eval_env, fsa, T)
    mp.options = options

    for i in range(50):

        mp.train_metapolicy(record=False, iters = 1)
        success, acc_reward = mp.evaluate_metapolicy(mp.mu)

        run.log({ 'metrics/evaluation/acc_reward': acc_reward,
                  'metrics/evaluation/success': success,
                  'metrics/evaluation/iter': i})

    wandb.finish()

if __name__ == "__main__":
    main()

