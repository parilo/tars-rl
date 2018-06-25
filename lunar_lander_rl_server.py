#!/usr/bin/env python

import os
import tensorflow as tf
import random
import numpy as np
from rl_server.rl_server import RLServer
from lunar_lander_model_dense import LunarLanderModelDense
from rl_server.algo.ddpg import DDPG
# from rl_server.algo.ddpg_prio_buf import DDPG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

history_len = 3

action_size = 2
observation_shapes = [(8,)]
state_shapes = [(history_len, 8,)]

critic_shapes = list(state_shapes)
critic_shapes.append((action_size,))
critic = LunarLanderModelDense(critic_shapes, 1, scope='critic')
actor = LunarLanderModelDense(state_shapes, action_size, scope='critic')

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
                       discount_factor=0.99,
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
                     gpu_id=0,
                     batch_size=512,
                     experience_replay_buffer_size=1000000,
                    #  use_prioritized_buffer=True,
                     use_prioritized_buffer=False,
                     train_every_nth=4,
                     history_length=history_len,
                     start_learning_after=5000,
                     target_networks_update_period=1000,
                     show_stats_period=100,
                     save_model_period=10000)

rl_server.start()
