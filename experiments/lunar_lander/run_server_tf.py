#!/usr/bin/env python

import sys

sys.path.append('../../')

import argparse
import random
import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.ddpg import DDPG
#from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG as DDPG
#from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG as DDPG
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks import *
from misc.experiment_config import ExperimentConfig

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller')
parser.add_argument(
    '--experiment_name',
    dest='experiment_name',
    type=str,
    default='experiment')
args = parser.parse_args()

config = ExperimentConfig(
    env_name='lunar_lander',
    experiment_name=args.experiment_name)

observation_shapes = [(config.obs_size,)]
state_shapes = [(config.history_len, config.obs_size,)]

critic = CriticNetwork(
    state_shapes[0], config.action_size, hiddens=[[256], [256]],
    activations=['relu', 'relu'], output_activation=None,
    action_insert_block=1, scope='critic')
#critic = QuantileCriticNetwork(
#    state_shapes[0], config.action_size, hiddens=[[256], [256]],
#    activations=['relu', 'relu'], output_activation=None,
#    num_atoms=128, action_insert_block=1, scope='critic')
#critic = CategoricalCriticNetwork(
#    state_shapes[0], config.action_size, hiddens=[[256], [256]],
#    activations=['relu', 'relu'], output_activation=None,
#    num_atoms=51, v=(-10., 10.), action_insert_block=1, scope='critic')

actor = ActorNetwork(
    state_shapes[0], config.action_size, hiddens=[[256], [256]],
    activations=['relu', 'relu'], output_activation='tanh',
    scope='actor')

agent_algorithm = DDPG(
    state_shapes=state_shapes,
    action_size=config.action_size,
    actor=actor,
    critic=critic,
    actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
    critic_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
    n_step=config.n_step,
    gradient_clip=1.0,
    discount_factor=config.gamma,
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
