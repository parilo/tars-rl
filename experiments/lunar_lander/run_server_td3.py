#!/usr/bin/env python

import sys
sys.path.append('../../')

import os
import argparse
import tensorflow as tf
import random
import numpy as np

from rl_server.rl_server import RLServer
from rl_server.algo.td3 import TD3
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

actor = ActorNetwork(state_shapes[0], C.action_size, hiddens=[[256, 256]],
                     activations=['relu'], output_activation='tanh',
                     layer_norm=False, noisy_layer=False, scope='actor')

critic1 = CriticNetwork(state_shapes[0], C.action_size, hiddens=[[256], [256]],
                        activations=['relu', 'relu'],
                        layer_norm=False, noisy_layer=False,
                        action_insert_block=1, scope='critic1')

critic2 = CriticNetwork(state_shapes[0], C.action_size, hiddens=[[256], [256]],
                        activations=['relu', 'relu'],
                        layer_norm=False, noisy_layer=False,
                        action_insert_block=1, scope='critic2')

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')

agent_algorithm = TD3(state_shapes=state_shapes,
                      action_size=C.action_size,
                      actor=actor,
                      critic1=critic1,
                      critic2=critic2,
                      actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      critic1_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      critic2_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      n_step=C.n_step,
                      gradient_clip=1.0,
                      action_noise_std=0.2,
                      action_noise_clip=0.5,
                      discount_factor=C.disc_factor,
                      target_actor_update_rate=5e-3,
                      target_critic1_update_rate=5e-3,
                      target_critic2_update_rate=5e-3)

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
                     target_critic_update_period=1,
                     target_actor_update_period=2,
                     show_stats_period=100,
                     save_model_period=10000,
                     init_port=C.port,
                     ckpt_path=C.path_to_ckpt)

info = agent_algorithm._get_info()
C.save_info(info)

rl_server.start()
