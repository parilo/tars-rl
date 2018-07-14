#!/usr/bin/env python

import sys
sys.path.append('../../')

import os
import json
import pickle
import argparse
import tensorflow as tf
import random
import numpy as np
from rl_server.rl_server import RLServer
from rl_server.networks.actor_networks import *
from rl_server.networks.critic_networks import *

def save_info(info, path):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + 'info.pkl', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

def load_info(path):
    with open(path + 'info.pkl', 'rb') as f:
        return pickle.load(f)

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

environment_name = 'lunar_lander'
experiment_config = json.load(open('config.txt'))

history_len = experiment_config['history_len']
frame_skip = experiment_config['frame_skip']
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

experiment_file = ''
for i in ['history_len', 'frame_skip', 'n_step', 'batch_size']:
    experiment_file = experiment_file + '-' + i + str(experiment_config[i])
if experiment_config['prio']:
    experiment_file = experiment_file + '-prio'
if experiment_config['sync']:
    experiment_file = experiment_file + '-sync'
else:
    experiment_file = experiment_file + '-async'

if args.experiment_name == 'experiment':
    path_to_experiment = 'results/' + experiment_file + '/'
else:
    path_to_experiment = 'results/' + args.experiment_name + '/'
path_to_ckpt = path_to_experiment + 'ckpt/'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from rl_server.algo.sac import SAC

observation_shapes = [(obs_size,)]
state_shapes = [(history_len, obs_size,)]

actor = GMMActorNetwork(state_shapes[0], action_size, hiddens=[[256, 256]],
                        activations=['relu'], output_activation='tanh',
                        layer_norm=False, noisy_layer=False,
                        num_components=4, scope='actor')

critic_v = CriticNetwork(state_shapes[0], action_size, hiddens=[[256, 256]],
                         activations=['relu'], output_activation=None,
                         layer_norm=False, noisy_layer=False,
                         action_insert_block=-1, scope='critic_v')

critic_q = CriticNetwork(state_shapes[0], action_size, hiddens=[[256], [256]],
                         activations=['relu', 'relu'], output_activation=None,
                         layer_norm=False, noisy_layer=False,
                         action_insert_block=1, scope='critic_q')

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')

agent_algorithm = SAC(state_shapes=state_shapes,
                      action_size=action_size,
                      actor=actor,
                      critic_v=critic_v,
                      critic_q=critic_q,
                      actor_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      critic_v_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      critic_q_optimizer=tf.train.AdamOptimizer(learning_rate=3e-4),
                      n_step=n_step,
                      gradient_clip=1.0,
                      discount_factor=disc_factor,
                      temperature=5e-3,
                      mu_and_sig_reg=0.,
                      target_critic_v_update_rate=1e-2)

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
                     train_every_nth=1,
                     history_length=history_len,
                     start_learning_after=500,
                     target_networks_update_period=1,
                     show_stats_period=100,
                     save_model_period=10000,
                     init_port=port,
                     ckpt_path=path_to_ckpt)

info = agent_algorithm._get_info()
save_info(info, path_to_experiment)

rl_server.start()
