import hydra 
from omegaconf import DictConfig, OmegaConf
import wandb 
from utils import seed_everything 
import gym 
import envs
from task_spec import load_fsa
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        sync_tensorboard=True, tags=["lof"], group="lof"
    )

    writer = SummaryWriter(f"/tmp/{run.name}")

    # Set seeds
    seed_everything(cfg.seed)

    env_cfg = dict(cfg.env)

    # Load the environments (train and eval)
    env_name = env_cfg.pop("gym_name")
    eval_name = env_cfg.pop("eval_name")
    env = gym.make(env_name)
    eval_env = gym.make(eval_name)
    fsa, T = load_fsa(cfg.fsa_name, env)

    eval_env = hydra.utils.call(config=env_cfg.pop("eval_env"), env=eval_env, fsa=fsa, fsa_init_state="u0", T=T)

    # Load the algorithm and run it
    policy = hydra.utils.call(config=cfg.algorithm, writer=writer, env=env, eval_env=eval_env, fsa=fsa, T=T)

    policy.learn_options()

    policy.train_metapolicy(record=False)

    policy.evaluate_metapolicy(log=True)

    # Create and save options and metapolicy
    os.makedirs(f"results/{run.name}/options")
    policy.save(f"results/{run.name}")

    writer.close()
    wandb.finish()

if __name__ == "__main__":

    main()