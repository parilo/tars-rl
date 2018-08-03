#!/usr/bin/env python

import sys
sys.path.append("../../")

import os
import json
import random
import numpy as np
from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks import *

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

environment_name = "l2run"
experiment_config = json.load(open("config.yml"))

history_len = experiment_config["history_len"]
n_step = experiment_config["n_step"]
obs_size = experiment_config["observation_size"]
action_size = experiment_config["action_size"]
use_prioritized_buffer = experiment_config["prio"]
batch_size = experiment_config["batch_size"]
prio = experiment_config["prio"]
use_synchronous_update = experiment_config["sync"]
port = experiment_config["port"]
gpu_id = experiment_config["gpu_id"]
disc_factor = experiment_config["disc_factor"]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from rl_server.tensorflow.algo.sac import SAC

observation_shapes = [(obs_size,)]
state_shapes = [(history_len, obs_size,)]

actor = GMMActorNetwork(state_shapes[0], action_size, hiddens=[[400, 300]],
                        activations=["relu"], output_activation="sigmoid",
                        num_components=4, scope="actor")

critic_v = CriticNetwork(state_shapes[0], action_size, hiddens=[[400, 300]],
                         activations=["relu"], output_activation=None,
                         action_insert_block=-1, scope="critic_v")

critic_q = CriticNetwork(state_shapes[0], action_size, hiddens=[[400, 300]],
                         activations=["relu"], output_activation=None,
                         action_insert_block=0, scope="critic_q")

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # "/path/to/checkpoint/model-4800000.ckpt")

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
                      temperature=1e-1,
                      target_critic_v_update_rate=0.01)

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
                     experience_replay_buffer_size=1000000,
                     use_prioritized_buffer=use_prioritized_buffer,
                     use_synchronous_update=use_synchronous_update,
                     train_every_nth=1,
                     history_length=history_len,
                     start_learning_after=5000,
                     target_networks_update_period=1,
                     show_stats_period=100,
                     save_model_period=10000,
                     init_port=port)

rl_server.start()
