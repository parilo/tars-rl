#!/usr/bin/env python

import sys
sys.path.append("../../")

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.algo_fabric import create_algorithm
from rl_server.tensorflow.algo.algo_ensemble import AlgoEnsemble
from rl_server.tensorflow.algo.base_algo import create_placeholders_n_algos_random_sample, create_placeholders
from misc.defaults import create_if_need, set_global_seeds
from misc.config import EnsembleConfig

set_global_seeds(4200)

############################# parse arguments #############################
parser = argparse.ArgumentParser(
    description="Run RL algorithm on RL server")
parser.add_argument(
    "--config",
    type=str,
    required=True)
parser.add_argument(
    "--logdir",
    type=str, 
    required=True)
args, unknown_args = parser.parse_known_args()

config = EnsembleConfig(args.config)

create_if_need(args.logdir)
observation_shapes, state_shapes, action_size = config.get_env_shapes()

# ############################# define algorithm ############################
# import tensorflow as tf
# from rl_server.tensorflow.networks.actor_networks_lstm import ActorNetwork
# from rl_server.tensorflow.networks.critic_networks_lstm import CriticNetwork
# from rl_server.tensorflow.algo.quantile_td3 import QuantileTD3

algo_config = config.algo_configs[0]

# actor_scope = "actor"
# algo_scope = "algorithm"
# 
# placeholders = create_placeholders(state_shapes, action_size)
# 
# actor_lr = placeholders[0]
# critic_lr = placeholders[1]
# 
# actor = ActorNetwork(
#     state_shape=state_shapes[0],
#     action_size=action_size,
#     **algo_config["actor"],
#     scope=actor_scope
# )
# 
# critic1 = CriticNetwork(
#     state_shape=state_shapes[0],
#     action_size=action_size,
#     **algo_config["critic"],
#     scope="critic1")
# critic2 = CriticNetwork(
#     state_shape=state_shapes[0],
#     action_size=action_size,
#     **algo_config["critic"],
#     scope="critic2")
# 
# agent_algorithm = QuantileTD3(
#     state_shapes=state_shapes,
#     action_size=action_size,
#     actor=actor,
#     critic1=critic1,
#     critic2=critic2,
#     actor_optimizer=tf.train.AdamOptimizer(
#         learning_rate=actor_lr),
#     critic1_optimizer=tf.train.AdamOptimizer(
#         learning_rate=critic_lr),
#     critic2_optimizer=tf.train.AdamOptimizer(
#         learning_rate=critic_lr),
#     **algo_config["algorithm"],
#     scope=algo_scope,
#     placeholders=placeholders,
#     actor_optim_schedule=algo_config["actor_optim"],
#     critic_optim_schedule=algo_config["critic_optim"],
#     training_schedule=algo_config["training"])

agent_algorithm = create_algorithm(
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    action_size=action_size,
    algo_config=algo_config,
    scope_postfix="lstm"
)

############################## run rl server ##############################
rl_server = RLServer(
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    ckpt_path=args.logdir,
    history_length=config.config["env"]["history_length"],
    **config.config["server"])
# rl_server.load_weights('/home/anton/devel/osim-rl-2018/experiments/prosthetics/logs/round2-activations-penalty/model-780000.ckpt')
rl_server.start()
