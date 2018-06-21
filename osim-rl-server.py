#!/usr/bin/env python

import os
import tensorflow as tf
import random
import numpy as np
from rl_server.rl_server import RLServer
from rl_server.osim_rl_model_dense import DenseNetwork
from rl_server.algo.ddpg import DDPG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

history_len = 3

# osim
#action_size = 18
#observation_shapes = [(41,)]
#state_shapes = [(41*3,)]

# pendulum
#action_size = 1
#observation_shapes = [(3,)]
#state_shapes = [(history_len*3,)]

# lunar lander
action_size = 2
observation_shapes = [(8,)]
state_shapes = [(history_len*8,)]

critic_shapes = list(state_shapes)
critic_shapes.append((action_size,))
critic = DenseNetwork(critic_shapes, 1, scope='critic')
actor = DenseNetwork(state_shapes, action_size, scope='actor')

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
                       target_actor_update_rate=1e-2,
                       target_critic_update_rate=1e-2)

rl_server = RLServer(num_clients=40,
                     action_size=action_size,
                     observation_shapes=observation_shapes,
                     state_shapes=state_shapes,
                     model_load_callback=model_load_callback,
                     agent_algorithm=agent_algorithm,
                     action_dtype=tf.float32,
                     is_actions_space_continuous=True,
                     gpu_id=0,
                     batch_size=128,
                     experience_replay_buffer_size=1000000,
                     train_every_nth=4,
                     history_length=history_len,
                     start_learning_after=5000,
                     target_networks_update_period=1,
                     show_stats_period=20,
                     save_model_period=10000)

rl_server.start()
