#!/usr/bin/env python

import sys
sys.path.append('../../')

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.ddpg_new import DDPG
from rl_server.tensorflow.algo.categorical_ddpg_new import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg_new import QuantileDDPG
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks import *
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
history_len = hparams["server"]["history_length"]

actor = ActorNetwork(
    state_shapes[0], action_size, hiddens=[[256], [256]],
    layer_norm=False, noisy_layer=False,
    activations=["relu", "relu"], output_activation="tanh",
    scope="actor")

if args.agent == "ddpg":
    critic = CriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation=None,
        action_insert_block=0, scope="critic")
    DDPG_algorithm = DDPG
elif args.agent == "categorical":
    critic = CategoricalCriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation=None,
        num_atoms=51, v=(-10., 10.),
        action_insert_block=0, scope="critic")
    DDPG_algorithm = CategoricalDDPG
elif args.agent == "quantile":
    critic = QuantileCriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation=None,
        num_atoms=128,
        action_insert_block=0, scope="critic")
    DDPG_algorithm = QuantileDDPG
else:
    raise NotImplementedError

agent_algorithm = DDPG_algorithm(
    state_shapes=state_shapes,
    action_size=action_size,
    actor=actor,
    critic=critic,
    actor_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    critic_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    n_step=hparams["algorithm"]["n_step"],
    actor_grad_val_clip=1.0,
    gamma=hparams["algorithm"]["gamma"],
    target_actor_update_rate=1.,
    target_critic_update_rate=1.)

rl_server = RLServer(
    num_clients=40,
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    batch_size=hparams["server"]["batch_size"],
    experience_replay_buffer_size=5000000,
    use_prioritized_buffer=hparams["server"]["use_prioritized_buffer"],
    use_synchronous_update=hparams["server"]["use_synchronous_update"],
    train_every_nth=1,
    history_length=history_len,
    start_learning_after=500,
    target_critic_update_period=500,
    target_actor_update_period=500,
    show_stats_period=100,
    save_model_period=1000000,
    init_port=hparams["server"]["init_port"],
    ckpt_path=args.logdir)

info = agent_algorithm._get_info()
rl_server.start()
