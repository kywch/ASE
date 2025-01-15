import yaml

import gym
import numpy as np
import torch

import isaacgym  # noqa
from isaacgym import gymapi
from isaacgym import gymutil

from ase.env.tasks.humanoid_amp_getup import HumanoidAMPGetup


# NOTE: See gymutil.parse_arguments for isaacgym args. Not using it here to simplify.
def parse_sim_params(args, cfg, sim_timestep=1.0 / 60.0, use_gpu=True):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = sim_timestep  # configs or args?

    # Use gpu and physx
    assert torch.cuda.is_available(), "CUDA is not available"
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    # NOTE: the default sim options are provided in cfg
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Use the default or provided arg params
    sim_params.physx.num_threads = args.physx_num_threads
    sim_params.physx.num_subscenes = args.physx_num_subscenes
    sim_params.num_client_threads = args.physx_num_client_threads

    return sim_params


def make_env(args, use_gpu=True):
    rl_device = "cpu"
    if use_gpu:
        assert torch.cuda.is_available(), "CUDA is not available"
        rl_device = "cuda:" + str(args.device_id)

    with open(args.env_cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    assert "env" in cfg, "env is not set in the config file"
    assert "sim" in cfg, "sim is not set in the config file"

    # Fill in the env config
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["motion_file"] = args.motion_file
    sim_params = parse_sim_params(args, cfg, use_gpu=use_gpu)

    # Use gpu and physx by default
    # NOTE: Start with training low-level controller, HumanoidAMPGetup
    task = HumanoidAMPGetup(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        device_type=rl_device,  # "cuda" if torch.cuda.is_available() and args.cuda else "cpu",
        device_id=args.device_id,
        headless=args.headless,
    )

    # add wrappers
    envs = VecTaskWrapper(task, rl_device, clip_observations=np.inf, clip_actions=1.0)
    print("num_envs: {:d}".format(envs.num_envs))
    print("num_actions: {:d}".format(envs.num_actions))
    print("num_obs: {:d}".format(envs.num_obs))
    print("num_states: {:d}".format(envs.num_states))

    envs = RecordEpisodeStatisticsTorch(envs, torch.device(rl_device))
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    return envs


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        if env_ids is None:
            self.episode_returns = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.returned_episode_returns = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.returned_episode_lengths = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
        else:
            self.episode_returns[env_ids] = 0
            self.episode_lengths[env_ids] = 0
            self.returned_episode_returns[env_ids] = 0
            self.returned_episode_lengths[env_ids] = 0
        return obs

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


# This wrapper combines VecTask, VecTaskPython, VecTaskPythonWrapper, RLGPUEnvWrapper
# Also does the action clipping
# CHECK ME: How about running mean norm on the observations?
class VecTaskWrapper:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_envs = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_obs = task.num_obs
        # print("task.num_obs",task.num_obs)
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = gym.spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = gym.spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )
        self.act_space = gym.spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device: ", rl_device)

        # RLGPU env wrapper
        self.use_global_obs = self.task.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()

        # AMP-related
        self._amp_obs_space = gym.spaces.Box(
            np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf
        )

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    # @property
    # def num_envs(self):
    #     return self.num_environments

    # @property
    # def num_acts(self):
    #     return self.num_actions

    # @property
    # def num_obs(self):
    #     return self.num_observations

    @property
    def gym(self):
        return self.task.gym

    @property
    def viewer(self):
        return self.task.viewer

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def _process_obs(self, obs):
        return torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        self.full_state["obs"] = self._process_obs(self.task.obs_buf)

        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def step(self, actions):
        # Action clipping
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        next_obs = self._process_obs(self.task.obs_buf)
        rewards = self.task.rew_buf.to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)
        infos = self.task.extras

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()
            return self.full_state, rewards, dones, infos
        else:
            return self.full_state["obs"], rewards, dones, infos
