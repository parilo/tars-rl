#!/usr/bin/env python

import tensorflow as tf
import random
import numpy as np
from rl_server.server.rl_server_api import RLServerAPI
from rl_server.rl_train_loop import RLTrainLoop
from rl_server.osim_rl_ddpg import OSimRLDDPG

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

num_clients = 4  # number of parallel simulators
action_size = 18
observation_shapes = [(41 * 3,)]

server_api = RLServerAPI(
    num_clients,
    observation_shapes
)

train_loop = RLTrainLoop(
    observation_shapes=observation_shapes,
    action_size=action_size,
    action_dtype=tf.float32,
    is_actions_space_continuous=True,
    batch_size=256,
    experience_replay_buffer_size=1000000,
    store_every_nth=1,
    train_every_nth=4,
    start_learning_after=5000,
    target_networks_update_period=500,
    show_stats_period=20,
    save_model_period=10000
)

agent_algorithm = OSimRLDDPG(
    observation_shapes=observation_shapes,
    action_size=action_size,
    discount_rate=0.99,
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    target_actor_update_rate=1.0,
    target_critic_update_rate=1.0
)

train_loop.set_algorithm(agent_algorithm)


def model_load_callback(sess, saver):
    pass

    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')


train_loop.init_vars(model_load_callback)

server_api.set_act_batch_callback(train_loop.act_batch)
server_api.set_store_exp_batch_callback(train_loop.store_exp_batch)
print('--- starting rl server')
server_api.start_server()
