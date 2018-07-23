#!/usr/bin/env python

import sys
sys.path.append('../../')

import os
import argparse
import tensorflow as tf
import random
import numpy as np

from rl_server.rl_server import RLServer
from rl_server.algo.prioritized_ddpg import PrioritizedDDPG as DDPG
from rl_server.networks.actor_networks import *
from rl_server.networks.critic_networks import *
from misc.experiment_config import ExperimentConfig

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--experiment_name',
                    dest='experiment_name',
                    type=str,
                    default='experiment')
args = parser.parse_args()

C = ExperimentConfig(env_name='lunar_lander', experiment_name=args.experiment_name)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(C.gpu_id)

observation_shapes = [(C.obs_size,)]
state_shapes = [(C.history_len, C.obs_size,)]

critic = CriticNetwork(state_shapes[0], C.action_size, hiddens=[[256], [256]],
                       activations=['relu', 'relu'], output_activation=None,
                       action_insert_block=1, scope='critic')

actor = ActorNetwork(state_shapes[0], C.action_size, hiddens=[[256], [256]],
                     activations=['relu', 'relu'], output_activation='tanh',
                     scope='actor')

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')

agent_algorithm = DDPG(state_shapes=state_shapes,
                       action_size=C.action_size,
                       actor=actor,
                       critic=critic,
                       actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                       critic_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                       n_step=C.n_step,
                       gradient_clip=1.0,
                       discount_factor=C.disc_factor,
                       target_actor_update_rate=1e-2,
                       target_critic_update_rate=1e-2)

rl_server = RLServer(num_clients=40,
                     action_size=C.action_size,
                     observation_shapes=observation_shapes,
                     state_shapes=state_shapes,
                     model_load_callback=model_load_callback,
                     agent_algorithm=agent_algorithm,
                     action_dtype=tf.float32,
                     is_actions_space_continuous=True,
                     gpu_id=C.gpu_id,
                     batch_size=C.batch_size,
                     experience_replay_buffer_size=100000,
                     use_prioritized_buffer=C.use_prioritized_buffer,
                     use_synchronous_update=C.use_synchronous_update,
                     train_every_nth=1,
                     history_length=C.history_len,
                     start_learning_after=500,
                     target_networks_update_period=1,
                     show_stats_period=100,
                     save_model_period=10000,
                     init_port=C.port,
                     ckpt_path=C.path_to_ckpt)

info = agent_algorithm._get_info()
C.save_info(info)

rl_server.start()

