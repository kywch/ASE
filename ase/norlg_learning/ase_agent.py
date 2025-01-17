import os
import time
import shutil
from datetime import datetime

import numpy as np
import torch
from torch import optim

from ase.norlg_learning.ase_players import CommonAgent

from ase.norlg_learning.utils import (
    to_torch,
    AverageMeter,
    AMPDataset,
    ExperienceBuffer,
    ReplayBuffer,
)

from tensorboardX import SummaryWriter


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def mean_list(val):
    return torch.mean(torch.stack(val))


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


def normalization_with_masks(values, masks):
    values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values


def get_mean_var_with_masks(values, masks):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = (((values_mask) ** 2) / sum_mask).sum() - ((values_mask / sum_mask).sum()) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)

    return values_mean, values_var


class ASEAgent(CommonAgent):
    def __init__(self, config, env):
        super().__init__(config, env)

        assert self._amp_input_mean_std is not None, "ampinput_mean_std must be set"

        self.use_action_masks = False  # config.get('use_action_masks', False)
        self.is_train = True  # config.get('is_train', True)
        self.ppo = True  # config['ppo']
        self.save_freq = config.get("save_frequency", 0)
        self.max_epochs = config.get("max_epochs", 0)

        self.num_actors = config["num_actors"]
        self.num_agents = self.env_info.get("agents", 1)

        self.algo_observer = config.get("algo_observer", None)
        self.network = config["network"]
        self.rewards_shaper = config["reward_shaper"]  # CHECK ME

        self.network_path = config.get("network_path", "./nn/")
        self.log_path = config.get("log_path", "runs/")

        # Reward weights
        self._task_reward_w = config.get("task_reward_w", 0.0)
        self._disc_reward_w = config.get("disc_reward_w", 0.0)
        self._enc_reward_w = config.get("enc_reward_w", 0.0)

        # PPO-related
        self.e_clip = config["e_clip"]
        self.clip_value = config["clip_value"]
        self.horizon_length = config["horizon_length"]
        self.normalize_advantage = config["normalize_advantage"]
        self.normalize_input = config["normalize_input"]
        self.grad_norm = config["grad_norm"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]

        # NOTE: ASE uses eps greedy. The policy does not learn when not using it.
        self._rand_action_probs = None

        # Loss weights
        self.critic_coef = config.get("critic_coef", 1.0)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.bounds_loss_coef = config.get("bounds_loss_coef", 0.0)
        self._disc_coef = config.get("disc_coef", 0.0)
        self._enc_coef = config.get("enc_coef", 0.0)
        # amp diversity loss
        self._amp_diversity_bonus = config.get("amp_diversity_bonus", 0.0)
        self._amp_diversity_tar = config.get("amp_diversity_tar", 0.0)

        # discriminator-related
        # self.bce_fn = torch.nn.BCEWithLogitsLoss()
        self._disc_logit_reg = config["disc_logit_reg"]
        self._disc_grad_penalty = config["disc_grad_penalty"]
        self._disc_weight_decay = config["disc_weight_decay"]
        self._disc_reward_scale = config["disc_reward_scale"]

        # self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = config["minibatch_size"]
        self.mini_epochs_num = config["mini_epochs"]
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert self.batch_size % self.minibatch_size == 0

        self._amp_batch_size = int(config["amp_batch_size"])
        self._amp_minibatch_size = int(config["amp_minibatch_size"])
        assert self._amp_minibatch_size <= self.minibatch_size

        self.obs = None
        self.last_lr = float(config["learning_rate"])
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0

        self.optimizer = optim.Adam(
            self.model.parameters(), self.last_lr, eps=1e-08, weight_decay=0.0
        )

        self.dataset = AMPDataset(self.batch_size, self.minibatch_size, self.device)

        self.done_indices = []

        # remove?
        self.mixed_precision = False  # self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.games_to_track = self.config.get("games_to_track", 100)
        self.game_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)

        # Check folders
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get("train_dir", "runs")

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_name = config["name"] + datetime.now().strftime("_%m-%d-%H-%M-%S")
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        self.summaries_dir = os.path.join(self.experiment_dir, "summaries")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        self.algo_observer.after_init(self)

    @property
    def task_env(self):
        return self.env.env.task

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.device)

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        self.experience_buffer.tensor_dict["next_obses"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["obses"]
        )
        self.experience_buffer.tensor_dict["next_values"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["values"]
        )

        batch_shape = self.experience_buffer.obs_base_shape
        # from ase.learning.amp_agent.py: AMPAgent._build_amp_buffers()
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self.amp_obs_shape, device=self.device
        )
        self.experience_buffer.tensor_dict["rand_action_mask"] = torch.zeros(
            batch_shape, dtype=torch.float32, device=self.device
        )

        amp_obs_demo_buffer_size = int(self.config["amp_obs_demo_buffer_size"])
        self._amp_obs_demo_buffer = ReplayBuffer(amp_obs_demo_buffer_size, self.device)

        self._amp_replay_keep_prob = self.config["amp_replay_keep_prob"]
        replay_buffer_size = int(self.config["amp_replay_buffer_size"])
        self._amp_replay_buffer = ReplayBuffer(replay_buffer_size, self.device)

        # from ase.learning.ase_agent.py: ASEAgent.init_tensors()
        self.experience_buffer.tensor_dict["ase_latents"] = torch.zeros(
            batch_shape + (self._latent_dim,), dtype=torch.float32, device=self.device
        )

        self._ase_latents = torch.zeros(
            (batch_shape[-1], self._latent_dim), dtype=torch.float32, device=self.device
        )
        self._latent_reset_steps = torch.zeros(
            batch_shape[-1], dtype=torch.int32, device=self.device
        )
        self._reset_latent_step_count()
        self._build_rand_action_probs()

        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + [
            "obses",
            "states",
            "dones",
            "next_obses",
            "amp_obs",
            "rand_action_mask",
            "ase_latents",
        ]

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)

        if env_ids is None:
            env_ids = self.all_env_ids

        if (len(env_ids) > 0):
            self._reset_latents(env_ids)
            self._reset_latent_step_count(env_ids)

        return obs

    def _reset_latent_step_count(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids

        self._latent_reset_steps[env_ids] = torch.randint_like(
            self._latent_reset_steps[env_ids],
            low=self._latent_steps_min,
            high=self._latent_steps_max,
        )

    def _build_rand_action_probs(self):
        num_envs = len(self.all_env_ids)
        self._rand_action_probs = 1.0 - torch.exp(10 * (self.all_env_ids / (num_envs - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0

        # NOTE: ASE uses eps greedy. The policy does not learn when not using it.
        # if not self._enable_eps_greedy:
        #     self._rand_action_probs[:] = 1.0

    def save(self, file_path):
        print("=> saving checkpoint '{}'".format(file_path))
        state_dict = self.get_model_weights()

        # Training state
        state_dict["epoch"] = self.epoch_num
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["frame"] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state_dict["last_mean_rewards"] = self.last_mean_rewards

        # Save the checkpoint
        torch.save(state_dict, file_path)

    def train(self):
        # CHECK ME: is resume_from necessary?
        # if self.resume_from != 'None':
        #     self.restore(self.resume_from)
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0  # frame = agent step

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = os.path.join(self.nn_dir, self.config["name"])

        self._init_amp_demo_buf()

        # MATCH xcxc debug -- init (norlg)
        # print("obs", self.obs.sum())

        epoch_num = 0
        while True:
            epoch_num += 1
            self.epoch_num = epoch_num

            # Collect data
            start_time = time.time()
            with torch.no_grad():
                batch_dict = self.play_steps()
            scaled_play_time = time.time() - start_time

            # Add amp obs
            # self._update_amp_demos()
            new_amp_obs_demo = self.task_env.fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": new_amp_obs_demo})

            num_obs_samples = batch_dict["amp_obs"].shape[0]
            amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
            batch_dict["amp_obs_demo"] = amp_obs_demo

            if self._amp_replay_buffer.get_total_count() == 0:
                batch_dict["amp_obs_replay"] = batch_dict["amp_obs"]
            else:
                amp_obs_replay = self._amp_replay_buffer.sample(num_obs_samples)["amp_obs"]
                batch_dict["amp_obs_replay"] = amp_obs_replay

            # Update the model
            update_time_start = time.time()
            train_info = None

            self.curr_frames = batch_dict.pop("played_frames")
            self.prepare_dataset(batch_dict)
            for _ in range(0, self.mini_epochs_num):
                for i in range(len(self.dataset)):
                    curr_train_info = self.calc_gradients(self.dataset[i])  # updating

                    if train_info is None:
                        train_info = dict()
                        for k, v in curr_train_info.items():
                            train_info[k] = [v]
                    else:
                        for k, v in curr_train_info.items():
                            train_info[k].append(v)

            for key in ["disc_rewards", "enc_rewards"]:
                train_info[key] = batch_dict[key]
            train_info["play_time"] = scaled_play_time
            train_info["update_time"] = time.time() - update_time_start

            self._store_replay_amp_obs(batch_dict["amp_obs"])

            # MATCH xcxc debug -- train epoch (norlg)
            # for k in ["amp_diversity_loss", "disc_loss", "disc_agent_logit", "disc_rewards", "enc_rewards"]:
            #     if isinstance(train_info[k], list):
            #         print(k, torch.stack(train_info[k]).sum())
            #     else:
            #         print(k, train_info[k].sum())

            # Log the stats
            sum_time = time.time() - start_time
            total_time += sum_time
            scaled_time = sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            mean_rewards = self.game_rewards.get_mean()
            if self.print_stats:
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                print(
                    "epoch_num:{}".format(epoch_num),
                    "mean_rewards:{}".format(mean_rewards),
                    f"fps step: {fps_step:.1f} fps total: {fps_total:.1f}",
                )

            frame = self.frame
            self.writer.add_scalar("rewards0/frame", mean_rewards, frame)
            self.writer.add_scalar("performance/total_fps", curr_frames / scaled_time, frame)
            self.writer.add_scalar("performance/step_fps", curr_frames / scaled_play_time, frame)
            self.writer.add_scalar("info/epochs", epoch_num, frame)
            self._log_train_info(train_info, frame)

            self.algo_observer.after_print_stats(frame, epoch_num, total_time)

            # save the checkpoint
            if self.save_freq > 0:
                if epoch_num % self.save_freq == 0:
                    self.save(model_output_file)

                    # save the intermediate checkpoints
                    int_model_output_file = model_output_file + "_" + str(epoch_num).zfill(8)
                    shutil.copyfile(model_output_file, int_model_output_file)

            if epoch_num > self.max_epochs:
                self.save(model_output_file)
                print("Reached the maximum number of epochs. Finshed training.")
                return self.last_mean_rewards, epoch_num

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))
        for _ in range(num_batches):
            curr_samples = self.task_env.fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": curr_samples})

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if buf_total_count > buf_size:
            keep_probs = to_torch(
                np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.device
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if amp_obs.shape[0] > buf_size:
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({"amp_obs": amp_obs})
        return

    def prepare_dataset(self, batch_dict):
        returns = batch_dict["returns"]
        values = batch_dict["values"]
        rand_action_mask = batch_dict["rand_action_mask"]

        advantages = torch.sum(returns - values, axis=1)
        if self.normalize_advantage:
            advantages = normalization_with_masks(advantages, rand_action_mask)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict["old_values"] = values
        dataset_dict["old_logp_actions"] = batch_dict["neglogpacs"]
        dataset_dict["advantages"] = advantages
        dataset_dict["returns"] = returns
        dataset_dict["actions"] = batch_dict["actions"]
        dataset_dict["obs"] = batch_dict["obses"]
        dataset_dict["mu"] = batch_dict["mus"]
        dataset_dict["sigma"] = batch_dict["sigmas"]
        # AMP & ASE
        dataset_dict["amp_obs"] = batch_dict["amp_obs"]
        dataset_dict["amp_obs_demo"] = batch_dict["amp_obs_demo"]
        dataset_dict["amp_obs_replay"] = batch_dict["amp_obs_replay"]
        dataset_dict["rand_action_mask"] = batch_dict["rand_action_mask"]
        dataset_dict["ase_latents"] = batch_dict["ase_latents"]

        self.dataset.update_values_dict(dataset_dict)

    def _log_train_info(self, train_info, frame):
        # Typical PPO train info
        self.writer.add_scalar("performance/update_time", train_info["update_time"], frame)
        self.writer.add_scalar("performance/play_time", train_info["play_time"], frame)
        self.writer.add_scalar("losses/a_loss", mean_list(train_info["actor_loss"]).item(), frame)
        self.writer.add_scalar("losses/c_loss", mean_list(train_info["critic_loss"]).item(), frame)

        self.writer.add_scalar("losses/bounds_loss", mean_list(train_info["b_loss"]).item(), frame)
        self.writer.add_scalar("losses/entropy", mean_list(train_info["entropy"]).item(), frame)
        self.writer.add_scalar(
            "info/last_lr", train_info["last_lr"][-1] * train_info["lr_mul"][-1], frame
        )
        self.writer.add_scalar("info/lr_mul", train_info["lr_mul"][-1], frame)
        self.writer.add_scalar("info/e_clip", self.e_clip * train_info["lr_mul"][-1], frame)
        self.writer.add_scalar(
            "info/clip_frac", mean_list(train_info["actor_clip_frac"]).item(), frame
        )
        self.writer.add_scalar("info/kl", mean_list(train_info["kl"]).item(), frame)

        # AMP & ASE info
        self.writer.add_scalar("losses/disc_loss", mean_list(train_info["disc_loss"]).item(), frame)
        self.writer.add_scalar("losses/enc_loss", mean_list(train_info["enc_loss"]).item(), frame)

        self.writer.add_scalar(
            "info/disc_agent_acc", mean_list(train_info["disc_agent_acc"]).item(), frame
        )
        self.writer.add_scalar(
            "info/disc_demo_acc", mean_list(train_info["disc_demo_acc"]).item(), frame
        )
        self.writer.add_scalar(
            "info/disc_agent_logit", mean_list(train_info["disc_agent_logit"]).item(), frame
        )
        self.writer.add_scalar(
            "info/disc_demo_logit", mean_list(train_info["disc_demo_logit"]).item(), frame
        )
        self.writer.add_scalar(
            "info/disc_grad_penalty", mean_list(train_info["disc_grad_penalty"]).item(), frame
        )
        self.writer.add_scalar(
            "info/disc_logit_loss", mean_list(train_info["disc_logit_loss"]).item(), frame
        )

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info["disc_rewards"])
        self.writer.add_scalar("info/disc_reward_mean", disc_reward_mean.item(), frame)
        self.writer.add_scalar("info/disc_reward_std", disc_reward_std.item(), frame)

        if "amp_diversity_loss" in train_info:
            self.writer.add_scalar(
                "losses/amp_diversity_loss",
                mean_list(train_info["amp_diversity_loss"]).item(),
                frame,
            )

        enc_reward_std, enc_reward_mean = torch.std_mean(train_info["enc_rewards"])
        self.writer.add_scalar("info/enc_reward_mean", enc_reward_mean.item(), frame)
        self.writer.add_scalar("info/enc_reward_std", enc_reward_std.item(), frame)

    #####################################################################

    def play_steps(self):
        self.set_eval()
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data("obses", n, self.obs)

            self._update_latents()

            res_dict = self.get_action_values(self.obs, self._ase_latents, self._rand_action_probs)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            # MATCH xcxc debug -- play steps, get_action_values (norlg)
            # print("obs", self.obs.sum())
            # print("ase latents", self._ase_latents.sum())
            # print("rand action probs", self._rand_action_probs.sum())
            # for k in res_dict.keys():
            #     try:
            #         print(k, res_dict[k].sum())
            #     except:
            #         pass

            """Stepping the environment"""
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            self.obs, rewards, self.dones, infos = self.env.step(res_dict["actions"])

            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)

            # No special reward shaping used. Remove.
            # shaped_rewards = self.rewards_shaper(rewards)
            shaped_rewards = rewards  # shape error

            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs)
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])
            self.experience_buffer.update_data("ase_latents", n, self._ase_latents)
            self.experience_buffer.update_data("rand_action_mask", n, res_dict["rand_action_mask"])

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs, self._ase_latents)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if self.task_env.viewer:
                self._amp_debug(infos, self._ase_latents)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        mb_ase_latents = self.experience_buffer.tensor_dict["ase_latents"]
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_ase_latents)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = swap_and_flatten01(v)

        # MATCH xcxc debug -- play steps (norlg)
        # for k in ["amp_obs", "ase_latents", "returns", "disc_rewards", "enc_rewards"]:
        #     print(k, batch_dict[k].sum())

        return batch_dict

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.task_env.progress_buf

        need_update = torch.any(new_latent_envs)
        if need_update:
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(
                self._latent_reset_steps[new_latent_env_ids],
                low=self._latent_steps_min,
                high=self._latent_steps_max,
            )
            if self.task_env.viewer:
                self._change_char_color(new_latent_env_ids)

    def get_action_values(self, obs_torch, ase_latents, rand_action_probs):
        processed_obs = self._preproc_obs(obs_torch)

        self.model.eval()
        with torch.no_grad():
            res_dict = self.model(
                {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": processed_obs,
                    "ase_latents": ase_latents,
                }
            )

        if self.normalize_value:
            res_dict["values"] = self.value_mean_std(res_dict["values"], True)

        # Implementing eps greedy
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict["actions"][det_action_mask] = res_dict["mus"][det_action_mask]
        res_dict["rand_action_mask"] = rand_action_mask

        return res_dict

    def _eval_critic(self, obs_torch, ase_latents):
        self.model.eval()
        processed_obs = self._preproc_obs(obs_torch)
        value = self.model.a2c_network.eval_critic(processed_obs, ase_latents)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards["disc_rewards"]
        enc_r = amp_rewards["enc_rewards"]
        combined_rewards = (
            self._task_reward_w * task_rewards
            + self._disc_reward_w * disc_r
            + self._enc_reward_w * enc_r
        )
        return combined_rewards

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    #####################################################################

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict["amp_obs"][0 : self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        # NOTE: The default _enable_enc_grad_penalty is False, so commenting it out
        # if (self._enable_enc_grad_penalty()):
        #     amp_obs.requires_grad_(True)

        amp_obs_replay = input_dict["amp_obs_replay"][0 : self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict["amp_obs_demo"][0 : self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        rand_action_mask = input_dict["rand_action_mask"]
        rand_action_sum = torch.sum(rand_action_mask)

        ase_latents = input_dict["ase_latents"]

        # lr = self.last_lr
        # kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
            "amp_obs": amp_obs,
            "amp_obs_replay": amp_obs_replay,
            "amp_obs_demo": amp_obs_demo,
            "ase_latents": ase_latents,
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]
            disc_agent_logit = res_dict["disc_agent_logit"]
            disc_agent_replay_logit = res_dict["disc_agent_replay_logit"]
            disc_demo_logit = res_dict["disc_demo_logit"]
            enc_pred = res_dict["enc_pred"]

            a_info = self._clip_policy_loss(
                old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip
            )
            a_loss = a_info["actor_loss"]
            a_clipped = a_info["actor_clipped"].float()

            c_info = self._clip_value_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(mu)

            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info["disc_loss"]

            enc_latents = batch_dict["ase_latents"][0 : self._amp_minibatch_size]
            enc_loss_mask = rand_action_mask[0 : self._amp_minibatch_size]
            enc_info = self._enc_loss(enc_pred, enc_latents, batch_dict["amp_obs"], enc_loss_mask)
            enc_loss = enc_info["enc_loss"]

            loss = (
                a_loss
                + self.critic_coef * c_loss
                - self.entropy_coef * entropy
                + self.bounds_loss_coef * b_loss
                + self._disc_coef * disc_loss
                + self._enc_coef * enc_loss
            )

            if self._amp_diversity_bonus > 0:
                diversity_loss = self._diversity_loss(
                    batch_dict["obs"], mu, batch_dict["ase_latents"]
                )
                diversity_loss = torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                loss += self._amp_diversity_bonus * diversity_loss
                a_info["amp_diversity_loss"] = diversity_loss

            a_info["actor_loss"] = a_loss
            a_info["actor_clip_frac"] = a_clip_frac
            c_info["critic_loss"] = c_loss

            self.optimizer.zero_grad()

        # TODO: remove self.scaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch)

        self.train_result = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        self.train_result.update(enc_info)

        return self.train_result

    def _clip_policy_loss(
        self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip
    ):
        # clipping the policy loss
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        return {"actor_loss": a_loss, "actor_clipped": clipped.detach()}

    def _clip_value_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        # clipping the value loss
        if clip_value:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -curr_e_clip, curr_e_clip
            )
            value_losses = (values - return_batch) ** 2
            value_losses_clipped = (value_pred_clipped - return_batch) ** 2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values) ** 2

        return {"critic_loss": c_loss}

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": disc_agent_logit.detach(),
            "disc_demo_logit": disc_demo_logit.detach(),
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    # NOTE: enc_obs is used for enc_grad_penalty, but not used by default
    def _enc_loss(self, enc_pred, ase_latent, enc_obs, loss_mask):
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        mask_sum = torch.sum(loss_mask)
        enc_err = enc_err.squeeze(-1)
        enc_loss = torch.sum(loss_mask * enc_err) / mask_sum
        enc_loss = torch.mean(enc_err)

        # NOTE: weight decay is 0 by default
        # if (self._enc_weight_decay != 0):
        #     enc_weights = self.model.a2c_network.get_enc_weights()
        #     enc_weights = torch.cat(enc_weights, dim=-1)
        #     enc_weight_decay = torch.sum(torch.square(enc_weights))
        #     enc_loss += self._enc_weight_decay * enc_weight_decay

        enc_info = {"enc_loss": enc_loss}
        return enc_info

        # NOTE: The default _enable_enc_grad_penalty is False, so commenting it out
        # if (self._enable_enc_grad_penalty()):
        #     enc_obs_grad = torch.autograd.grad(enc_err, enc_obs, grad_outputs=torch.ones_like(enc_err),
        #                                        create_graph=True, retain_graph=True, only_inputs=True)
        #     enc_obs_grad = enc_obs_grad[0]
        #     enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
        #     #enc_grad_penalty = torch.sum(loss_mask * enc_obs_grad) / mask_sum
        #     enc_grad_penalty = torch.mean(enc_obs_grad)
        #     enc_loss += self._enc_grad_penalty * enc_grad_penalty
        #     enc_info['enc_grad_penalty'] = enc_grad_penalty.detach()

    def _diversity_loss(self, obs, action_params, ase_latents):
        assert self.model.a2c_network.is_continuous

        n = obs.shape[0]
        assert n == action_params.shape[0]

        new_z = self._sample_latents(n)
        mu, sigma = self.model.a2c_network.eval_actor(obs=obs, ase_latents=new_z)

        clipped_action_params = torch.clamp(action_params, -1.0, 1.0)
        clipped_mu = torch.clamp(mu, -1.0, 1.0)

        a_diff = clipped_action_params - clipped_mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * ase_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(self._amp_diversity_tar - diversity_bonus)

        return diversity_loss
