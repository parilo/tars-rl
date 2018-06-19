#!/usr/bin/env python

import tensorflow as tf
import random
import numpy as np
from rl_server.rl_server import RLServer
from rl_server.osim_rl_model_dense import DenseNetwork
from rl_server.algo.ddpg import DDPG

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# osim
# action_size = 18
# observation_shapes = [(41,)]
# state_shapes = [(41*3,)]

# pendulum
action_size = 1
observation_shapes = [(3,)]
state_shapes = [(3*3,)]

critic_shapes = list(state_shapes)
critic_shapes.append((action_size,))
critic = DenseNetwork(critic_shapes, 1, scope='critic')
actor = DenseNetwork(state_shapes, action_size, scope='actor')

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # '/path/to/checkpoint/model-4800000.ckpt')

agent_algorithm = DDPG(
    observation_shapes=state_shapes,
    action_size=action_size,
    actor=actor,
    critic=critic,
    actor_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    critic_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    discount_rate=0.99,
    target_actor_update_rate=1.0,
    target_critic_update_rate=1.0)

rl_server = RLServer(
    num_clients=40,  # number of parallel simulators
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
    train_every_nth=4,
    history_length=3,
    start_learning_after=5000,
    target_networks_update_period=500,
    show_stats_period=20,
    save_model_period=10000)

rl_server.start()
