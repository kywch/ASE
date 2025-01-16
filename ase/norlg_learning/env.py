import numpy as np

from ase.env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from ase.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from ase.utils.config import parse_sim_params


def create_rlgpu_env(args, cfg, cfg_train, **kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    if rl_device == "cpu":
        sim_params.use_gpu_pipeline = False
        sim_params.physx.use_gpu = False

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    # NOTE: Start with training low-level controller, HumanoidAMPGetup
    try:
        task = HumanoidAMPGetup(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
        )

    except NameError as e:
        print(e)

    env = VecTaskPythonWrapper(
        task,
        rl_device,
        cfg_train.get("clip_observations", np.inf),
        cfg_train.get("clip_actions", 1.0),
    )

    print("num_envs: {:d}".format(env.num_envs))
    print("num_actions: {:d}".format(env.num_actions))
    print("num_obs: {:d}".format(env.num_obs))
    print("num_states: {:d}".format(env.num_states))

    return env


def get_env_info(env):
    result_shapes = {}
    result_shapes["observation_space"] = env.observation_space
    result_shapes["action_space"] = env.action_space
    result_shapes["amp_observation_space"] = env.amp_observation_space
    result_shapes["agents"] = 1
    result_shapes["value_size"] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes["agents"] = env.get_number_of_agents()
    """
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    """
    if hasattr(env, "value_size"):
        result_shapes["value_size"] = env.value_size
    print(result_shapes)
    return result_shapes


class RLGPUEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.use_global_obs = self.env.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

        # method mapping
        self.get_number_of_agents = self.env.get_number_of_agents

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def amp_observation_space(self):
        return self.env.amp_observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def value_size(self):
        return 1  # self.env.value_size

    @property
    def num_envs(self):
        return self.env.num_envs

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["amp_observation_space"] = self.env.amp_observation_space

        if self.use_global_obs:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info
