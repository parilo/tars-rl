#!/usr/bin/env python

import sys
sys.path.append("../../")

import os
import argparse
import random
import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG as DDPG
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks import *
from misc.experiment_config import ExperimentConfig

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser(description="Train or test neural net motor controller")
parser.add_argument("--experiment_name",
                    dest="experiment_name",
                    type=str,
                    default="experiment")
args = parser.parse_args()

C = ExperimentConfig(env_name="prosthetics_new", experiment_name=args.experiment_name)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(C.gpu_id)

observation_shapes = [(C.obs_size,)]
state_shapes = [(C.history_len, C.obs_size,)]

critic = CategoricalCriticNetwork(state_shapes[0], C.action_size, hiddens=[[400], [300]],
                                  layer_norm=False, noisy_layer=False,
                                  activations=["relu", "relu"], output_activation=None,
                                  action_insert_block=0, num_atoms=101, v=(-100., 900.), scope="critic")

actor = ActorNetwork(state_shapes[0], C.action_size, hiddens=[[400], [300]],
                     layer_norm=False, noisy_layer=False,
                     activations=["relu", "relu"], output_activation="sigmoid",
                     scope="actor")

def model_load_callback(sess, saver):
    pass
    # examples of loading checkpoint
    # saver.restore(sess,
    # "/path/to/checkpoint/model-4800000.ckpt")

agent_algorithm = DDPG(state_shapes=state_shapes,
                       action_size=C.action_size,
                       actor=actor,
                       critic=critic,
                       actor_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
                       critic_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
                       n_step=C.n_step,
                       gradient_clip=1.0,
                       discount_factor=C.gamma,
                       target_actor_update_rate=1.0,
                       target_critic_update_rate=1.0)

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
                     experience_replay_buffer_size=1000000,
                     use_prioritized_buffer=C.use_prioritized_buffer,
                     use_synchronous_update=C.use_synchronous_update,
                     train_every_nth=2,
                     history_length=C.history_len,
                     start_learning_after=5000,
                     target_networks_update_period=500,
                     show_stats_period=10,
                     save_model_period=10000,
                     init_port=C.port)

info = agent_algorithm._get_info()
C.save_info(info)

rl_server.start()
