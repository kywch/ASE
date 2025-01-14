import gym
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def shape_whc_to_cwh(shape):
    # if len(shape) == 2:
    #    return (shape[1], shape[0])
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])

    return shape


# (-1, 1) -> (low, high)
def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class DefaultAlgoObserver:
    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = AverageMeter(1, self.algo.games_to_track).to(self.algo.device)
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return
        if not isinstance(infos, dict) and len(infos) > 0 and isinstance(infos[0], dict):
            done_indices = done_indices.cpu()
            for ind in done_indices:
                ind = ind.item()
                if len(infos) <= ind // self.algo.num_agents:
                    continue
                info = infos[ind // self.algo.num_agents]
                game_res = None
                if "battle_won" in info:
                    game_res = info["battle_won"]
                if "scores" in info:
                    game_res = info["scores"]

                if game_res is not None:
                    self.game_scores.update(
                        torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device)
                    )

    def after_steps(self):
        pass

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0 and self.writer is not None:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class DefaultRewardsShaper:
    def __init__(
        self, scale_value=1, shift_value=0, min_val=-np.inf, max_val=np.inf, is_torch=True
    ):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.is_torch = is_torch

    def __call__(self, reward):
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        if self.is_torch:
            import torch

            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            if mask is not None:
                raise NotImplementedError
                # mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = (
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
                )
            )

        # change shape
        # if self.per_channel:
        #     if len(self.insize) == 3:
        #         current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
        #         current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
        #     if len(self.insize) == 2:
        #         current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
        #         current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
        #     if len(self.insize) == 1:
        #         current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
        #         current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        # else:
        current_mean = self.running_mean
        current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


# class RunningMeanStdObs(nn.Module):
#     def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
#         assert(insize is dict)
#         super().__init__()
#         self.running_mean_std = nn.ModuleDict({
#             k : RunningMeanStd(v, epsilon, per_channel, norm_only) for k,v in insize.items()
#         })

#     def forward(self, input, unnorm=False):
#         res = {k : self.running_mean_std(v, unnorm) for k,v in input.items()}
#         return res


class ExperienceBuffer:
    """
    More generalized than replay buffers.
    Implemented for on-policy algos
    """

    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get("agents", 1)
        self.action_space = env_info["action_space"]

        self.num_actors = algo_info["num_actors"]
        self.horizon_length = algo_info["horizon_length"]
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)

        self.actions_shape = (self.action_space.shape[0],)
        self.actions_num = self.action_space.shape[0]
        self.is_continuous = True

        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        self.tensor_dict["obses"] = self._create_tensor_from_space(
            env_info["observation_space"], obs_base_shape
        )

        val_space = gym.spaces.Box(low=0, high=1, shape=(env_info.get("value_size", 1),))
        self.tensor_dict["rewards"] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict["values"] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict["neglogpacs"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32), obs_base_shape
        )
        self.tensor_dict["dones"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8), obs_base_shape
        )
        self.tensor_dict["actions"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            obs_base_shape,
        )
        self.tensor_dict["mus"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            obs_base_shape,
        )
        self.tensor_dict["sigmas"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            obs_base_shape,
        )

    def _init_from_aux_dict(self, tensor_dict):
        obs_base_shape = self.obs_base_shape
        for k, v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(v), dtype=np.float32), obs_base_shape
            )

    def _create_tensor_from_space(self, space, base_shape):
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)

        raise ValueError(f"Unsupported space type: {type(space)}")

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        else:
            self.tensor_dict[name][index, :] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if type(v) is dict:
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict


class AMPDataset(Dataset):
    def __init__(self, batch_size, minibatch_size, device):
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size

        self.special_names = ["rnn_states"]
        self._idx_buf = torch.randperm(batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._get_item(idx)

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        raise NotImplementedError()

        # start = self.last_range[0]
        # end = self.last_range[1]
        # self.values_dict["mu"][start:end] = mu
        # self.values_dict["sigma"][start:end] = sigma

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]

        if end >= self.batch_size:
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
        return


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self._head = 0
        self._total_count = 0
        self._buffer_size = buffer_size
        self._device = device
        self._data_buf = None
        self._sample_idx = torch.randperm(buffer_size)
        self._sample_head = 0

    def reset(self):
        self._head = 0
        self._total_count = 0
        self._reset_sample_idx()

    def get_buffer_size(self):
        return self._buffer_size

    def get_total_count(self):
        return self._total_count

    def store(self, data_dict):
        if self._data_buf is None:
            self._init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        buffer_size = self.get_buffer_size()
        assert n <= buffer_size

        for key, curr_buf in self._data_buf.items():
            curr_n = data_dict[key].shape[0]
            assert n == curr_n

            store_n = min(curr_n, buffer_size - self._head)
            curr_buf[self._head : (self._head + store_n)] = data_dict[key][:store_n]

            remainder = n - store_n
            if remainder > 0:
                curr_buf[0:remainder] = data_dict[key][store_n:]

        self._head = (self._head + n) % buffer_size
        self._total_count += n

    def sample(self, n):
        total_count = self.get_total_count()
        buffer_size = self.get_buffer_size()

        idx = torch.arange(self._sample_head, self._sample_head + n)
        idx = idx % buffer_size
        rand_idx = self._sample_idx[idx]
        if total_count < buffer_size:
            rand_idx = rand_idx % self._head

        samples = dict()
        for k, v in self._data_buf.items():
            samples[k] = v[rand_idx]

        self._sample_head += n
        if self._sample_head >= buffer_size:
            self._reset_sample_idx()

        return samples

    def _reset_sample_idx(self):
        buffer_size = self.get_buffer_size()
        self._sample_idx[:] = torch.randperm(buffer_size)
        self._sample_head = 0

    def _init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()
        self._data_buf = dict()

        for k, v in data_dict.items():
            v_shape = v.shape[1:]
            self._data_buf[k] = torch.zeros((buffer_size,) + v_shape, device=self._device)
