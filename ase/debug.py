import copy
import numpy as np

# isaacgym must be imported before torch
import isaacgym
import torch

from rl_games.common import env_configurations

# RLG
from ase.learning import ase_players as rlg_ase_players

# NO RLG
from ase.norlg_learning.env import create_rlgpu_env
from ase.norlg_learning.ase_players import ASEPlayer
from ase.norlg_learning.network import ASENetworkBuilder, ASEModelBuilder
from ase.norlg_learning.utils import DefaultRewardsShaper
from ase.utils.config import set_np_formatting, get_args, load_cfg

RUN_RLG = False


# Replace rlgames' torch_runner and factories
class Runner:
    def __init__(self, env_creator, algo_observer=None):
        self.env_creator = env_creator
        self.algo_observer = algo_observer
        # torch.backends.cudnn.benchmark = True  # make non-deterministic

    def reset(self):
        pass

    def load(self, yaml_conf):
        self.default_config = yaml_conf["params"]
        self.load_config(copy.deepcopy(self.default_config))

        # if 'experiment_config' in yaml_conf:
        #     self.exp_config = yaml_conf['experiment_config']

    def load_config(self, params):  # params = cfg_train
        self.seed = params.get("seed", None)

        self.algo_params = params["algo"]
        self.algo_name = self.algo_params["name"]
        self.load_check_point = params["load_checkpoint"]
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            print("Found checkpoint")
            print(params["load_path"])
            self.load_path = params["load_path"]

        # self.model is actually model builder, not the model itself
        self.model = self.make_model_builder(params)
        self.config = copy.deepcopy(params["config"])

        self.config["reward_shaper"] = DefaultRewardsShaper(**self.config["reward_shaper"])
        self.config["network"] = self.model

    def make_model_builder(self, params):
        # if RUN_RLG:
        #     network_builder = rlg_ase_network_builder.ASEBuilder()
        #     network_builder.load(params["network"])
        #     model_builder = rlg_ase_models.ModelASEContinuous(network_builder)

        # else:
        network_builder = ASENetworkBuilder()
        network_builder.load(params["network"])
        model_builder = ASEModelBuilder(network_builder)
        return model_builder

    def run(self, args):
        if "checkpoint" in args and args["checkpoint"] is not None:
            if len(args["checkpoint"]) > 0:
                self.load_path = args["checkpoint"]

        if args["train"]:
            raise NotImplementedError
            # self.run_train()

        elif args["play"]:
            print("Started to play")
            player = self.create_player()
            player.restore(self.load_path)
            player.run()

        else:
            raise ValueError(f"Unknown command: {args}")

    def create_player(self):
        if RUN_RLG:
            return rlg_ase_players.ASEPlayer(self.config)
        else:
            return ASEPlayer(self.config, self.env_creator)


if __name__ == "__main__":
    set_np_formatting()
    args = get_args()

    # Manually provide args
    args.seed = 1
    args.train = False
    args.play = True
    args.task = "HumanoidAMPGetup"
    args.num_envs = 1
    args.cfg_env = "ase/data/cfg/humanoid_ase_sword_shield_getup.yaml"
    args.cfg_train = "ase/data/cfg/train/rlg/ase_humanoid.yaml"
    # args.motion_file = "ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml"
    args.motion_file = "ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_Jump_Motion.npy"
    args.checkpoint = "ase/data/models/ase_llc_reallusion_sword_shield.pth"

    # Load config
    cfg, cfg_train, logdir = load_cfg(args)

    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file

    # Create default directories for weights and statistics
    cfg_train["params"]["config"]["train_dir"] = args.output_path

    env_creator = lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs)
    env_configurations.register("rlgpu", {"env_creator": env_creator, "vecenv_type": "RLGPU"})

    vargs = vars(args)

    runner = Runner(env_creator)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)
