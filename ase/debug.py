import copy
import numpy as np

# isaacgym must be imported before torch
import isaacgym
import torch

from rl_games.common import env_configurations

# RLG
from ase.learning import ase_players as rlg_ase_players
from ase.learning import ase_agent as rlg_ase_agent
import ase.learning.rlgpu

# NO RLG
from ase.norlg_learning.env import create_rlgpu_env, RLGPUEnvWrapper
from ase.norlg_learning.ase_players import ASEPlayer
from ase.norlg_learning.ase_agent import ASEAgent
from ase.norlg_learning.network import ASENetworkBuilder, ASEModelBuilder
from ase.norlg_learning.utils import DefaultRewardsShaper, DefaultAlgoObserver
from ase.utils.config import set_np_formatting, get_args, load_cfg


RUN_RLG = True
RUN_EVAL = False


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
        network_builder = ASENetworkBuilder()
        network_builder.load(params["network"])
        model_builder = ASEModelBuilder(network_builder)
        return model_builder

    def run(self, args):
        if "checkpoint" in args and args["checkpoint"] is not None:
            if len(args["checkpoint"]) > 0:
                self.load_path = args["checkpoint"]

        if args["train"]:
            self.run_train()

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

    def run_train(self):
        print("Started to train")
        self.reset()
        self.load_config(self.default_config)

        if self.algo_observer is None:
            self.algo_observer = DefaultAlgoObserver()
        self.config["algo_observer"] = self.algo_observer

        if RUN_RLG:
            self.config["features"] = {"observer": self.algo_observer}
            agent = rlg_ase_agent.ASEAgent(base_name="run", config=self.config)

        else:
            vec_env = self.env_creator()
            vec_env = RLGPUEnvWrapper(vec_env)
            agent = ASEAgent(self.config, vec_env)

        if self.load_check_point and (self.load_path is not None):
            agent.restore(self.load_path)

        agent.train()


if __name__ == "__main__":
    set_np_formatting()
    args = get_args()

    # Manually provide args
    args.seed = 1
    args.task = "HumanoidAMPGetup"
    args.cfg_env = "ase/data/cfg/humanoid_ase_sword_shield_getup.yaml"
    args.cfg_train = "ase/data/cfg/train/rlg/ase_humanoid.yaml"

    if RUN_EVAL:
        args.test = True
        args.num_envs = 1
        # args.motion_file = (
        #     "ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml"
        # )
        args.motion_file = "ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_Jump_Motion.npy"
        args.checkpoint = "ase/data/models/ase_llc_reallusion_sword_shield.pth"
        # args.checkpoint = "test/test3_8.pth"
        # args.checkpoint = "ase/data/models/test_3hr.pth"

    else:
        args.motion_file = (
            "ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml"
        )
        # args.motion_file = "ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_Jump_Motion.npy"
        args.headless = True

    # Set the correct mode
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    vargs = vars(args)

    # Load config
    cfg, cfg_train, logdir = load_cfg(args)

    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file

    # Create default directories for weights and statistics
    cfg_train["params"]["config"]["train_dir"] = args.output_path

    env_creator = lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs)
    env_configurations.register("rlgpu", {"env_creator": env_creator, "vecenv_type": "RLGPU"})

    runner = Runner(env_creator)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)
