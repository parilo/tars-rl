#!/usr/bin/env python

import sys
sys.path.append("../")

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.quantile_td3 import QuantileTD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork
from misc.defaults import default_parse_fn, create_if_need, set_global_seeds

set_global_seeds(42)

############################# parse arguments #############################

parser = argparse.ArgumentParser(
    description="Run RL algorithm on RL server")
parser.add_argument(
    "--agent", 
    default="ddpg",
    choices=["ddpg", "categorical", "quantile", "td3", "sac", "quant_td3"])
parser.add_argument(
    "--hparams",
    type=str, 
    required=True)
parser.add_argument(
    "--logdir",
    type=str, 
    required=True)
args, unknown_args = parser.parse_known_args()
create_if_need(args.logdir)
args, hparams = default_parse_fn(args, unknown_args)
observation_shapes = [(hparams["env"]["obs_size"],)]
history_len = hparams["server"]["history_length"]
state_shapes = [(history_len, hparams["env"]["obs_size"],)]
action_size = hparams["env"]["action_size"]

############################# define algorithm ############################

if args.agent != "sac" and args.agent != "td3" and args.agent != "quant_td3":

    actor = ActorNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["actor"],
        scope="actor")

    if args.agent == "ddpg":
        critic = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic")
        DDPG_algorithm = DDPG
    elif args.agent == "categorical":
        critic = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            num_atoms=51,
            v=(-10., 10.),
            scope="critic")
        DDPG_algorithm = CategoricalDDPG
    elif args.agent == "quantile":
        critic = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic")
        DDPG_algorithm = QuantileDDPG
    else:
        raise NotImplementedError

    agent_algorithm = DDPG_algorithm(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic=critic,
        actor_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["actor_optim"]["lr"]),
        critic_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        **hparams["algorithm"])

#######################################################################
########################## Soft Actor-Critic ##########################
#######################################################################

elif args.agent == "sac":

    actor = GaussActorNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["actor"],
        scope="actor")

    critic_v = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic_v"],
        scope="critic_v")

    critic_q1 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic_q"],
        scope="critic_q1")

    critic_q2 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic_q"],
        scope="critic_q2")

    agent_algorithm = SAC(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic_q1=critic_q1,
        critic_q2=critic_q2,
        critic_v=critic_v,
        actor_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["actor_optim"]["lr"]),
        critic_q1_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        critic_q2_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        critic_v_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        **hparams["algorithm"])

#######################################################################
########################## Twin Delayed DDPG ##########################
#######################################################################

elif args.agent == "td3":

    actor = ActorNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["actor"],
        scope="actor")
   
    critic1 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic"],
        scope="critic1")
    critic2 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic"],
        scope="critic2")

    agent_algorithm = TD3(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["actor_optim"]["lr"]),
        critic1_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        critic2_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        **hparams["algorithm"])
    
#######################################################################
########################## Twin Delayed DDPG ##########################
#######################################################################

elif args.agent == "quant_td3":

    actor = ActorNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["actor"],
        scope="actor")
   
    critic1 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic"],
        scope="critic1")
    critic2 = CriticNetwork(
        state_shape=state_shapes[0],
        action_size=action_size,
        **hparams["critic"],
        scope="critic2")

    agent_algorithm = QuantileTD3(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["actor_optim"]["lr"]),
        critic1_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        critic2_optimizer=tf.train.AdamOptimizer(
            learning_rate=hparams["critic_optim"]["lr"]),
        **hparams["algorithm"])

#######################################################################
############################ run rl server ############################
#######################################################################

rl_server = RLServer(
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    ckpt_path=args.logdir,
    **hparams["server"])
rl_server.start()
