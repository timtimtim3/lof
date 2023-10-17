import hydra 
from omegaconf import DictConfig, OmegaConf
import wandb 
from utils import seed_everything 
import gym 
from task_spec import load_fsa
from envs.wrapper import DeliveryAutomatonEnv
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        sync_tensorboard=True,
    )

    writer = SummaryWriter(f"/tmp/{run.name}")

    # Set seeds
    seed_everything(cfg.seed)

    env_config = dict(cfg.env)

    env_name = env_config.pop("gym_name")
    eval_name = env_config.pop("eval_name")
    
    env = gym.make(env_name)

    eval_env = gym.make(eval_name)
    fsa, T = load_fsa(cfg.fsa_name, env)
    eval_env = DeliveryAutomatonEnv(eval_env, fsa,  "u0", T)

    policy = hydra.utils.call(config=cfg.algorithm, writer=writer, env=env, eval_env=eval_env, fsa=fsa, T=T)


    policy._learn_options()
    policy.train_metapolicy(record=True)

    writer.close()
    wandb.finish()

if __name__ == "__main__":

    main()