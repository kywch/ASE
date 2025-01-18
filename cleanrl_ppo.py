import os
import random
import time
import yaml
from dataclasses import dataclass

import isaacgym  # noqa

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from ase.cleanrl.env import make_env


DEBUG = False


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_id: int = 0
    """the gpu id to use"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ase"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Env/sim-specific arguments
    env_id: str = "HumanoidAMPGetup"
    """the id of the environment"""
    env_cfg_file: str = "ase/data/cfg/humanoid_ase_sword_shield_getup.yaml"
    """the path to the environment configuration file"""
    # motion_file: str = "ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml"
    motion_file: str = "ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_Jump_Motion.npy"  # DEBUG
    """the path to the motion file"""
    headless: bool = True
    """whether to run the environment in headless mode"""
    physx_num_threads: int = 4
    """the number of cores used by PhysX"""
    physx_num_subscenes: int = 0
    """the number of PhysX subscenes to simulate in parallel"""
    physx_num_client_threads: int = 0
    """the number of client threads that process env slices"""

    # NOTE: train config file -- ase/data/cfg/train/rlg/ase_humanoid.yaml
    # Fix the network params, move the train params to here

    # PPO-specific arguments
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-5
    """the learning rate of the optimizer"""
    num_envs: int = 4096 if not DEBUG else 32
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0005
    """coefficient of the entropy"""
    vf_coef: float = 5
    """coefficient of the value function"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    reward_scaler: float = 1
    """the scale factor applied to the reward during training"""
    # record_video_step_frequency: int = 1464
    # """the frequency at which to record the videos"""
    save_freq: int = 600 if not DEBUG else 5
    """the frequency at which to save the checkpoints"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("warn")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"
    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    save_dir = f"runs/{run_name}"
    writer = SummaryWriter(save_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    seed_everything(args)

    # env setup
    envs = make_env(args, use_gpu=args.cuda and not DEBUG)
    device = envs.device

    print()