import numpy as np
import torch
import torch.nn as nn
from functools import reduce

from rl_server.torch.networks.sequential import SequentialNet


def fanin_init(layer):
    if isinstance(layer, nn.Linear):
        fanin = layer.weight.data.shape[0]
        v = 1. / np.sqrt(fanin)
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)


def out_init(layer):
    if isinstance(layer, nn.Linear):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)


class Actor(nn.Module):
    def __init__(
            self, observation_shape, n_action, hiddens, layer_fn,
            activation_fn=torch.nn.ReLU, norm_fn=None, bias=True,
            out_activation=nn.Sigmoid):
        super().__init__()

        n_observation = reduce(lambda x, y: x*y, observation_shape)

        self.feature_net = SequentialNet(
            hiddens=[n_observation] + hiddens,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias)
        self.policy_net = SequentialNet(
            hiddens=[hiddens[-1], n_action],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None, bias=True)

        self.feature_net.apply(fanin_init)
        self.policy_net.apply(out_init)

    def forward(self, observation):
        observation = observation.view(observation.shape[0], -1)
        x = observation
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)
        return x

    def get_info(self):
        info = {}
        return info


class Critic(nn.Module):
    def __init__(
            self, observation_shape, n_action, hiddens, layer_fn,
            concat_at=1, n_atoms=1,
            activation_fn=torch.nn.ReLU, norm_fn=None, bias=True,
            out_activation=None):
        super().__init__()
        self.n_atoms = n_atoms

        n_observation = reduce(lambda x, y: x * y, observation_shape)

        if concat_at > 0:
            self.observation_net = SequentialNet(
                hiddens=[n_observation] + hiddens[0:concat_at],
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias)
            self.feature_net = SequentialNet(
                hiddens=[hiddens[concat_at] + n_action] + hiddens[concat_at:],
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias)
        else:
            self.observation_net = None
            self.feature_net = SequentialNet(
                hiddens=[n_observation + n_action] + hiddens,
                layer_fn=layer_fn,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                bias=bias)

        self.value_net = SequentialNet(
            hiddens=[hiddens[-1], n_atoms],
            layer_fn=nn.Linear,
            activation_fn=out_activation,
            norm_fn=None, bias=True)

        if self.observation_net is not None:
            self.observation_net.apply(fanin_init)
        self.feature_net.apply(fanin_init)
        self.value_net.apply(out_init)

    def forward(self, observation, action):
        observation = observation.view(observation.shape[0], -1)
        if self.observation_net is not None:
            observation = self.observation_net(observation)
        x = torch.cat((observation, action), dim=1)
        x = self.feature_net.forward(x)
        x = self.value_net.forward(x)
        return x

    def get_info(self):
        info = {}
        return info
