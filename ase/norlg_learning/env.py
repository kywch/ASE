import numpy as np

from ase.env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from ase.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from ase.utils.config import parse_sim_params


def create_rlgpu_env(args, cfg, cfg_train, **kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

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
