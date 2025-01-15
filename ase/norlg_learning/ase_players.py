# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time

import gym
import numpy as np
import torch

from ase.norlg_learning.env import get_env_info
from ase.norlg_learning.utils import rescale_actions, RunningMeanStd, shape_whc_to_cwh


class CommonAgent:
    def __init__(self, config, env):
        # from rl_games.common.player import BasePlayer
        # BasePlayer.__init__(self, config)
        self.config = config
        # self.env_name = self.config['env_name']
        # self.env_config = self.config.get('env_config', {})
        self.env_info = self.config.get("env_info")
        self.clip_actions = config.get("clip_actions", True)

        self.env = env
        if self.env_info is None:
            # self.env = env_creator(**self.env_config)  # self.create_env()
            self.env_info = get_env_info(self.env)

        self.value_size = self.env_info.get("value_size", 1)
        self.action_space = self.env_info["action_space"]
        self.num_agents = self.env_info["agents"]

        self.observation_space = self.env_info["observation_space"]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.amp_obs_shape = self.env_info["amp_observation_space"].shape

        self.states = None
        self.player_config = self.config.get("player", {})
        self.batch_size = 1
        # self.has_central_value = self.config.get('central_value_config') is not None
        self.render_env = self.player_config.get("render", False)
        self.games_num = self.player_config.get("games_num", 15)
        self.is_determenistic = self.player_config.get("determenistic", True)
        self.print_stats = self.player_config.get("print_stats", True)
        self.render_sleep = self.player_config.get("render_sleep", 0.002)
        self.max_steps = 108000 // 4

        # TODO: check device in config. For now, there is no device nor device_name
        self.use_cuda = True
        self.device_name = self.config.get("device_name", "cuda")
        self.device = torch.device(self.device_name)

        # from rl_games.algos_torch.players import PpoPlayerContinuous
        self.network = config["network"]

        self._setup_action_space()
        self.mask = [False]

        self._latent_dim = config["latent_dim"]
        self.normalize_input = self.config.get("normalize_input", False)
        self.normalize_value = self.config.get("normalize_value", False)
        self._normalize_amp_input = config.get("normalize_amp_input", True)
        self._build_model()

        # ase latent-related
        if hasattr(self, "env"):
            num_envs = self.task_env.num_envs
        else:
            num_envs = self.env_info["num_envs"]
        self.all_env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self._ase_latents = torch.zeros(
            (num_envs, self._latent_dim), dtype=torch.float32, device=self.device
        )

        self._latent_steps_min = config.get("latent_steps_min", np.inf)
        self._latent_steps_max = config.get("latent_steps_max", np.inf)

        # Used in training, _amp_debug
        self._disc_reward_scale = config["disc_reward_scale"]
        self._enc_reward_scale = config["enc_reward_scale"]

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

    def _build_model(self):
        obs_shape = shape_whc_to_cwh(self.obs_shape)
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            # 'num_seqs' : self.num_agents  # used for rnn, so not needed
            "amp_input_shape": self.amp_obs_shape,
            "ase_latent_shape": (self._latent_dim,),
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.is_rnn = self.model.is_rnn()
        assert not self.is_rnn, "ASE policy does not use RNN"

        self.running_mean_std = (
            RunningMeanStd(obs_shape).to(self.device) if self.normalize_input else None
        )
        self.value_mean_std = RunningMeanStd((1,)).to(self.device) if self.normalize_value else None
        self._amp_input_mean_std = (
            RunningMeanStd(self.amp_obs_shape).to(self.device)
            if self._normalize_amp_input
            else None
        )
        self.set_eval()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

    def get_model_weights(self):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        if self.normalize_input:
            state_dict["running_mean_std"] = self.running_mean_std.state_dict()
        if self.normalize_value:
            state_dict["value_mean_std"] = self.value_mean_std.state_dict()
        if self._normalize_amp_input:
            state_dict["amp_input_mean_std"] = self._amp_input_mean_std.state_dict()
        return state_dict

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(state_dict["running_mean_std"])
        if self.normalize_value and "value_mean_std" in state_dict:
            self.value_mean_std.load_state_dict(state_dict["value_mean_std"])
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(state_dict["amp_input_mean_std"])

    def restore(self, file_path):
        assert os.path.exists(file_path), "Checkpoint file does not exist"
        print("=> loading checkpoint '{}'".format(file_path))
        state_dict = torch.load(file_path)
        self.set_model_weights(state_dict)

    def env_reset(self, env_ids=None):
        obs_torch = self.env.reset(env_ids)
        # obs is already in torch
        return obs_torch

    def _preproc_obs(self, obs_batch):
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    @property
    def task_env(self):
        raise NotImplementedError

    def _change_char_color(self, env_ids):
        if self.task_env.viewer is None:
            return

        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.task_env.set_char_color(rand_col, env_ids)

    def _reset_latents(self, done_env_ids=None):
        if done_env_ids is None:
            done_env_ids = self.all_env_ids

        rand_vals = self._sample_latents(len(done_env_ids))
        self._ase_latents[done_env_ids] = rand_vals
        self._change_char_color(done_env_ids)

    def _sample_latents(self, num):
        return self.model.a2c_network.sample_latents(num)

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {"disc_rewards": disc_r, "enc_rewards": enc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def _calc_enc_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale
        return enc_r

    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _amp_debug(self, info, ase_latents=None):
        if ase_latents is None:
            ase_latents = self._ase_latents

        with torch.no_grad():
            amp_obs = info["amp_obs"]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards["disc_rewards"]
            enc_reward = amp_rewards["enc_rewards"]

        disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
        disc_reward = disc_reward.cpu().numpy()[0, 0]
        enc_reward = enc_reward.cpu().numpy()[0, 0]
        print("disc_pred: ", disc_pred, disc_reward, enc_reward)


class ASEPlayer(CommonAgent):
    def __init__(self, config, env_creator):
        env_config = config.get("env_config", {})
        env = env_creator(**env_config)
        super().__init__(config, env)

    @property
    def task_env(self):
        return self.env.task

    def get_batch_size(self, obses):
        obs_shape = self.obs_shape
        assert len(obses.size()) > len(obs_shape), "obses must be batched"
        return obses.size()[0]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        games_played = 0
        self._reset_latent_step_count()

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_torch = self.env_reset()
            batch_size = self.get_batch_size(obs_torch)

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            done_indices = []

            # NOTE: check if there is a separate env for playing the dataset
            if False:  # self.env.task.play_dataset:
                # play dataset
                while True:
                    for t in range(self.env.task.max_episode_length):
                        self.env.task.play_dataset_step(t)

            else:
                # inference
                for _ in range(self.max_steps):
                    obs_torch = self.env_reset(done_indices)

                    action = self.get_action(obs_torch, is_determenistic)

                    """Stepping the environment"""
                    obs_torch, r, done, info = self.env.step(action)

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        if self.print_stats:
                            print(
                                "games_played:",
                                games_played,
                                "reward:",
                                cur_rewards / done_count,
                                "steps:",
                                cur_steps / done_count,
                            )

                        if batch_size // self.num_agents == 1 or games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(n_games, "games played. Done.")
        return

    def get_action(self, obs_torch, is_determenistic=False):
        self._update_latents()

        # obs = obs_dict['obs']
        # if len(obs.size()) == len(self.obs_shape):
        #     obs = obs.unsqueeze(0)
        obs_torch = self._preproc_obs(obs_torch)
        ase_latents = self._ase_latents

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs_torch,
            # 'rnn_states' : self.states,
            "ase_latents": ase_latents,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = None  # res_dict['rnn_states']

        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        # current_action = current_action.detach()

        return rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0)
        )

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        self._reset_latents(env_ids)
        return obs

    def _update_latents(self):
        if self._latent_step_count <= 0:
            self._reset_latents()
            self._reset_latent_step_count()

            if self.task_env.viewer:
                print("Sampling new amp latents------------------------------")
                self._change_char_color(self.all_env_ids)
        else:
            self._latent_step_count -= 1

    def _reset_latent_step_count(self):
        self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)

    def _post_step(self, info):
        if self.task_env.viewer:
            self._amp_debug(info)
