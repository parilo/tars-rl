#!/usr/bin/env python

import sys
sys.path.append('../../')

import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from rl_server.torch.rl_server import RLServer
from rl_server.torch.networks.agents import Actor, Critic
from rl_server.torch.algorithms.ddpg import DDPG
from rl_server.torch.algorithms.quantile_ddpg import QuantileDDPG
from rl_server.torch.algorithms.categorical_ddpg import CategoricalDDPG
from rl_server.torch.algorithms.td3 import TD3
from misc.defaults import default_parse_fn, create_if_need, set_global_seeds

set_global_seeds(42)

parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller')
parser.add_argument(
    "--agent", default="ddpg",
    choices=["ddpg", "categorical", "quantile", "td3"])
parser.add_argument(
    "--hparams",
    type=str, required=True)
parser.add_argument(
    "--logdir",
    type=str, required=True)
args, unknown_args = parser.parse_known_args()

create_if_need(args.logdir)
args, hparams = default_parse_fn(args, unknown_args)

observation_shapes = [(hparams["env"]["obs_size"],)]
state_shapes = [
    (hparams["server"]["history_length"], hparams["env"]["obs_size"],)]
action_size = hparams["env"]["action_size"]

actor = Actor(
    observation_shape=state_shapes[0], n_action=action_size,
    **hparams["actor"])
critic = Critic(
        observation_shape=state_shapes[0], n_action=action_size,
        **hparams["critic"], out_activation=None)

if args.agent == "ddpg":
    DDPG_algorithm = DDPG
elif args.agent == "categorical":
    critic = Critic(
        observation_shape=state_shapes[0], n_action=action_size,
        **hparams["critic"], out_activation=lambda: nn.Softmax(dim=1))
    DDPG_algorithm = CategoricalDDPG
elif args.agent == "quantile":
    DDPG_algorithm = QuantileDDPG
elif args.agent == "td3":
    critic2 = Critic(
        observation_shape=state_shapes[0], n_action=action_size,
        **hparams["critic"], out_activation=None)
    DDPG_algorithm = TD3
else:
    raise NotImplementedError

if args.agent == "td3":
    agent_algorithm = DDPG_algorithm(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic=critic,
        critic2=critic2,
        actor_optimizer=torch.optim.Adam(
            actor.parameters(), **hparams["actor_optim"]),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), **hparams["critic_optim"]),
        critic2_optimizer=torch.optim.Adam(
            critic2.parameters(), **hparams["critic_optim"]),
        **hparams["algorithm"],)
else:
    agent_algorithm = DDPG_algorithm(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(
            actor.parameters(), **hparams["actor_optim"]),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), **hparams["critic_optim"]),
        **hparams["algorithm"])

rl_server = RLServer(
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    logdir=args.logdir, **hparams["server"])

info = agent_algorithm._get_info()

rl_server.start()
