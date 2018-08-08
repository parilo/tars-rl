#!/usr/bin/env python

import sys
sys.path.append('../../')

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork
from misc.defaults import default_parse_fn, create_if_need, set_global_seeds

set_global_seeds(42)

parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller')
parser.add_argument(
    "--agent", default="ddpg",
    choices=["ddpg", "categorical", "quantile", "td3", "sac"])
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

if args.agent != "sac" and args.agent != "td3":
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
        critic = CriticNetwork(
            state_shapes[0], action_size, hiddens=[[256], [256]],
            layer_norm=False, noisy_layer=False,
            activations=["relu", "relu"], output_activation=None,
            num_atoms=51, v=(-10., 10.),
            action_insert_block=0, scope="critic")
        DDPG_algorithm = CategoricalDDPG
    elif args.agent == "quantile":
        critic = CriticNetwork(
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
        target_actor_update_rate=1e-2,
        target_critic_update_rate=1e-2)
elif args.agent == "sac":
    actor = GMMActorNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        activations=["relu", "relu"], output_activation="tanh",
        layer_norm=False, noisy_layer=False,
        num_components=4, scope="actor")
    critic_v = CriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        activations=["relu", "relu"], output_activation=None,
        layer_norm=False, noisy_layer=False,
        action_insert_block=-1, scope="critic_v")
    critic_q = CriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        activations=["relu", "relu"], output_activation=None,
        layer_norm=False, noisy_layer=False,
        action_insert_block=0, scope="critic_q")
    agent_algorithm = SAC(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic_v=critic_v,
        critic_q=critic_q,
        actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        critic_v_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        critic_q_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        n_step=hparams["algorithm"]["n_step"],
        actor_grad_val_clip=None,
        gamma=hparams["algorithm"]["gamma"],
        reward_scale=200,
        target_critic_update_rate=1e-2)
elif args.agent == "td3":
    actor = ActorNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation="tanh",
        scope="actor")
    critic1 = CriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation=None,
        action_insert_block=0, scope="critic1")
    critic2 = CriticNetwork(
        state_shapes[0], action_size, hiddens=[[256], [256]],
        layer_norm=False, noisy_layer=False,
        activations=["relu", "relu"], output_activation=None,
        action_insert_block=0, scope="critic2")
    agent_algorithm = TD3(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        critic1_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        critic2_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
        n_step=hparams["algorithm"]["n_step"],
        actor_grad_val_clip=None,
        gamma=hparams["algorithm"]["gamma"],
        target_actor_update_rate=1e-2,
        target_critic_update_rate=1e-2)
        
rl_server = RLServer(
    num_clients=40,
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    batch_size=hparams["server"]["batch_size"],
    experience_replay_buffer_size=100000,
    use_prioritized_buffer=hparams["server"]["use_prioritized_buffer"],
    use_synchronous_update=hparams["server"]["use_synchronous_update"],
    train_every_nth=1,
    history_length=history_len,
    start_learning_after=500,
    target_critic_update_period=1,
    target_actor_update_period=1,
    show_stats_period=100,
    save_model_period=1000000,
    init_port=hparams["server"]["init_port"],
    ckpt_path=args.logdir)

info = agent_algorithm._get_info()
rl_server.start()
