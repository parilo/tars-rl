#!/usr/bin/env python

import os
import json
import tensorflow as tf
import random
import numpy as np
from rl_server.rl_server import RLServer
from rl_server.networks.actor_networks import ActorNetwork
from rl_server.networks.critic_networks import *

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

environment_name = 'prosthetics_new'
experiment_config = json.load(open('configs/' + environment_name + '.txt'))

history_len = experiment_config['history_len']
n_step = experiment_config['n_step']
obs_size = experiment_config['observation_size']
action_size = experiment_config['action_size']
use_prioritized_buffer = experiment_config['prio']
batch_size = experiment_config['batch_size']
prio = experiment_config['prio']
use_synchronous_update = experiment_config['sync']
port = experiment_config['port']
gpu_id = experiment_config['gpu_id']
disc_factor = experiment_config['disc_factor']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

if prio:
    from rl_server.algo.prioritized_ddpg import DDPG
else:
    from rl_server.algo.categorical_ddpg import DDPG

observation_shapes = [(obs_size,)]
state_shapes = [(history_len, obs_size,)]

critic = CategoricalCriticNetwork(state_shapes[0], action_size, hiddens=[[400], [300]],
                       activations=['relu', 'relu'], output_activation=None,
                       action_insert_block=0, num_atoms=51, v=(-5., 5.), scope='critic')

actor = ActorNetwork(state_shapes[0], action_size, hiddens=[[400], [300]],
                     activations=['relu', 'tanh'], output_activation='tanh',
                     scope='actor')

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')

agent_algorithm = DDPG(state_shapes=state_shapes,
                       action_size=action_size,
                       actor=actor,
                       critic=critic,
                       actor_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
                       critic_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
                       n_step=n_step,
                       gradient_clip=1.0,
                       discount_factor=disc_factor,
                       target_actor_update_rate=1.0,
                       target_critic_update_rate=1.0)

rl_server = RLServer(num_clients=40,
                     action_size=action_size,
                     observation_shapes=observation_shapes,
                     state_shapes=state_shapes,
                     model_load_callback=model_load_callback,
                     agent_algorithm=agent_algorithm,
                     action_dtype=tf.float32,
                     is_actions_space_continuous=True,
                     gpu_id=gpu_id,
                     batch_size=batch_size,
                     experience_replay_buffer_size=100000,
                     use_prioritized_buffer=use_prioritized_buffer,
                     use_synchronous_update=use_synchronous_update,
                     train_every_nth=4,
                     history_length=history_len,
                     start_learning_after=5000,
                     target_networks_update_period=1000,
                     show_stats_period=100,
                     save_model_period=10000,
                     init_port=port)

rl_server.start()

