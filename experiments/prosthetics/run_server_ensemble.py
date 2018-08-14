#!/usr/bin/env python

import sys
sys.path.append("../../")

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.algo_fabric import create_algorithm
from rl_server.tensorflow.algo.algo_ensemble import AlgoEnsemble
from rl_server.tensorflow.algo.base_algo import create_placeholders
from misc.defaults import default_parse_fn, create_if_need, set_global_seeds

set_global_seeds(42)

############################# parse arguments #############################
parser = argparse.ArgumentParser(
    description="Run RL algorithm on RL server")
parser.add_argument(
    "--agent", 
    default="ddpg",
    choices=["ddpg", "categorical", "quantile", "td3", "sac"])
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
placeholders = create_placeholders(state_shapes, action_size)
agent_algorithms = [create_algorithm(args.agent, hparams, placeholders, i) for i in range(hparams["ensemble"]["num_of_algorithms"])]
algo_ensemble = AlgoEnsemble(agent_algorithms, placeholders)

############################## run rl server ##############################
rl_server = RLServer(
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=algo_ensemble,
    ckpt_path=args.logdir,
    **hparams["server"])
rl_server.start()
