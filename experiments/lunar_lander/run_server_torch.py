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
from misc.experiment_config import ExperimentConfig

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller')
parser.add_argument(
    '--experiment_name',
    dest='experiment_name',
    type=str,
    default='experiment')
parser.add_argument(
    "--agent", default="ddpg",
    choices=["ddpg", "categorical", "quantile", "td3"])
args = parser.parse_args()

config = ExperimentConfig(
    env_name='lunar_lander',
    experiment_name=args.experiment_name)

observation_shapes = [(config.obs_size,)]
state_shapes = [(config.history_len, config.obs_size,)]


actor = Actor(
    observation_shape=state_shapes[0], n_action=config.action_size,
    hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
    bias=False, activation_fn=nn.ReLU, out_activation=nn.Tanh)

if args.agent == "ddpg":
    critic = Critic(
        observation_shape=state_shapes[0], n_action=config.action_size,
        hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
        bias=False, activation_fn=nn.ReLU,
        concat_at=1, n_atoms=1, out_activation=None)
    DDPG_algorith = DDPG
elif args.agent == "categorical":
    critic = Critic(
        observation_shape=state_shapes[0], n_action=config.action_size,
        hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
        bias=False, activation_fn=nn.ReLU,
        concat_at=1, n_atoms=51, out_activation=lambda: nn.Softmax(dim=1))
    DDPG_algorith = CategoricalDDPG
elif args.agent == "quantile":
    critic = Critic(
        observation_shape=state_shapes[0], n_action=config.action_size,
        hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
        bias=False, activation_fn=nn.ReLU,
        concat_at=1, n_atoms=128, out_activation=None)
    DDPG_algorith = QuantileDDPG
elif args.agent == "td3":
    critic = Critic(
        observation_shape=state_shapes[0], n_action=config.action_size,
        hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
        bias=False, activation_fn=nn.ReLU,
        concat_at=1, n_atoms=1, out_activation=None)
    critic2 = Critic(
        observation_shape=state_shapes[0], n_action=config.action_size,
        hiddens=[256, 256], layer_fn=nn.Linear, norm_fn=None,
        bias=False, activation_fn=nn.ReLU,
        concat_at=1, n_atoms=1, out_activation=None)
    DDPG_algorith = TD3
else:
    raise NotImplementedError


if args.agent == "td3":
    agent_algorithm = DDPG_algorith(
        state_shapes=state_shapes,
        action_size=config.action_size,
        actor=actor,
        critic=critic,
        critic2=critic2,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        critic2_optimizer=torch.optim.Adam(critic2.parameters(), lr=3e-4),
        n_step=config.n_step,
        actor_grad_clip=1.0,
        gamma=config.gamma,
        target_actor_update_rate=1e-2,
        target_critic_update_rate=1e-2)
else:
    agent_algorithm = DDPG_algorith(
        state_shapes=state_shapes,
        action_size=config.action_size,
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        n_step=config.n_step,
        actor_grad_clip=1.0,
        gamma=config.gamma,
        target_actor_update_rate=1e-2,
        target_critic_update_rate=1e-2)

rl_server = RLServer(
    num_clients=40,
    action_size=config.action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    batch_size=config.batch_size,
    experience_replay_buffer_size=100000,
    use_prioritized_buffer=config.use_prioritized_buffer,
    use_synchronous_update=config.use_synchronous_update,
    train_every_nth=1,
    history_length=config.history_len,
    start_learning_after=500,
    target_critic_update_period=1,
    target_actor_update_period=1,
    show_stats_period=100,
    save_model_period=10000,
    init_port=config.port,
    ckpt_path=config.path_to_ckpt)

info = agent_algorithm._get_info()
config.save_info(info)

rl_server.start()