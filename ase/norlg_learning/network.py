import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ASENetworkBuilder:
    def __init__(self, **kwargs):
        self.params = None

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        assert self.params is not None, "params is not set"
        net = ASENetworkBuilder.Network(self.params, **kwargs)
        return net

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            amp_input_shape = kwargs.get('amp_input_shape')
            self._ase_latent_shape = kwargs.get('ase_latent_shape')

            super().__init__()

            self.units = params["mlp"]["units"]  # = [1024, 1024, 512]
            self.space_config = params["space"]["continuous"]

            # Actor
            style_units = [512, 256]
            style_dim = self._ase_latent_shape[-1]
            self.actor_mlp = AMPStyleCatNet1(obs_size=input_shape[-1],
                                             ase_latent_size=style_dim,
                                             units=self.units,
                                             style_units=style_units,
                                             style_dim=style_dim)

            actor_out_size = self.actor_mlp.get_out_size()
            self.mu = nn.Linear(actor_out_size, actions_num)

            if self.space_config["learn_sigma"]:
                self.sigma = nn.Linear(actor_out_size, actions_num)
                nn.init.constant_(self.sigma.weight, self.space_config["sigma_init"]["val"])
            else:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                nn.init.constant_(self.sigma, self.space_config["sigma_init"]["val"])

            # Critic
            self.critic_mlp = AMPMLPNet(obs_size=input_shape[-1],
                                        ase_latent_size=style_dim,
                                        units=self.units)

            critic_out_size = self.critic_mlp.get_out_size()
            self.value = nn.Linear(critic_out_size, 1)

            # Discriminator and encoder
            self._enc_mlp = self._disc_mlp = nn.Sequential(
                layer_init(nn.Linear(amp_input_shape[-1], self.units[0])),
                nn.ReLU(),
                layer_init(nn.Linear(self.units[0], self.units[1])),
                nn.ReLU(),
                layer_init(nn.Linear(self.units[1], self.units[2])),
                nn.ReLU(),
            )
            self._disc_logits = layer_init(nn.Linear(self.units[-1], 1))
            self._enc = layer_init(nn.Linear(self.units[-1], self._ase_latent_shape[-1]))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            ase_latents = obs_dict['ase_latents']
            states = None  # obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs, ase_latents)
            value = self.eval_critic(obs, ase_latents)

            output = actor_outputs + (value, states)
            return output

        def eval_critic(self, obs, ase_latents, use_hidden_latents=False):
            c_out = self.critic_mlp(obs, ase_latents, use_hidden_latents)
            return self.value(c_out)

        def eval_actor(self, obs, ase_latents, use_hidden_latents=False):
            a_out = self.actor_mlp(obs, ase_latents, use_hidden_latents)
            mu = self.mu(a_out)
            if self.space_config['fixed_sigma']:
                sigma = self.sigma
            else:
                sigma = self.sigma(a_out)

            return mu, sigma

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def eval_enc(self, amp_obs):
            enc_mlp_out = self._enc_mlp(amp_obs)
            enc_output = self._enc(enc_mlp_out)
            enc_output = nn.functional.normalize(enc_output, dim=-1)
            return enc_output

        def sample_latents(self, n):
            device = next(self._enc.parameters()).device
            z = torch.normal(torch.zeros([n, self._ase_latent_shape[-1]], device=device))
            z = nn.functional.normalize(z, dim=-1)
            return z


class ASEModelBuilder:
    def __init__(self, network_builder):
        self.network_builder = network_builder

    def build(self, config):
        net = self.network_builder.build(None, **config)
        for name, _ in net.named_parameters():
            print(name)
        return ASEModelBuilder.Network(net)

    # from rl_games.algos_torch.models import ModelA2CContinuousLogStd
    # class Network(ModelA2CContinuousLogStd.Network):
    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return False  # self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return None  # self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": value,
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )


class AMPMLPNet(nn.Module):
    def __init__(self, obs_size, ase_latent_size, units):
        super().__init__()

        input_size = obs_size + ase_latent_size
        print('build amp mlp net:', input_size)
        
        self._units = units  # [1024, 1024, 512]
        self._mlp = nn.Sequential(
            layer_init(nn.Linear(input_size, self._units[0])),
            nn.ReLU(),
            layer_init(nn.Linear(self._units[0], self._units[1])),
            nn.ReLU(),
            layer_init(nn.Linear(self._units[1], self._units[2])),
            nn.ReLU(),
        )

    def forward(self, obs, latent, skip_style):
        inputs = [obs, latent]
        input = torch.cat(inputs, dim=-1)
        output = self._mlp(input)
        return output

    def get_out_size(self):
        out_size = self._units[-1]
        return out_size


class AMPStyleCatNet1(nn.Module):
    def __init__(self, obs_size, ase_latent_size, units,
                 style_units, style_dim):
        super().__init__()

        print('build amp style cat net:', obs_size, ase_latent_size)

        self._style_mlp = nn.Sequential(  # style_units = [512, 256]
            layer_init(nn.Linear(ase_latent_size, style_units[0])),
            nn.ReLU(),
            layer_init(nn.Linear(style_units[0], style_units[1])),
            nn.ReLU(),
        )
        self._style_dense = nn.Linear(style_units[1], style_dim)
        self._style_activation = nn.Tanh()  # torch.tanh

        self._units = units
        self._dense_layers = nn.ModuleList([
            layer_init(nn.Linear(obs_size + style_dim, self._units[0])),
            layer_init(nn.Linear(self._units[0], self._units[1])),
            layer_init(nn.Linear(self._units[1], self._units[2])),
        ])
        self._activation = nn.ReLU()

    def forward(self, obs, latent, skip_style):
        if (skip_style):
            style = latent
        else:
            style = self.eval_style(latent)

        h = torch.cat([obs, style], dim=-1)

        # NOTE: too fragmented ... but leave it for now, so that we can use the existing checkpoints
        for i in range(len(self._dense_layers)):
            curr_dense = self._dense_layers[i]
            h = curr_dense(h)
            h = self._activation(h)

        return h

    def eval_style(self, latent):
        # NOTE: too fragmented ... but leave it for now, so that we can use the existing checkpoints
        style_h = self._style_mlp(latent)
        style = self._style_dense(style_h)
        style = self._style_activation(style)
        return style

    def get_out_size(self):
        out_size = self._units[-1]
        return out_size
