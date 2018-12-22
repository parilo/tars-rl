#!/usr/bin/env python

import sys
sys.path.append("../../")

import argparse
import random

import numpy as np

# from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.algo_fabric import create_algorithm
from misc.common import create_if_need, set_global_seeds, parse_server_args
from misc.config import ExperimentConfig

set_global_seeds(42)
args = parse_server_args()

exp_config = ExperimentConfig(args.config)
create_if_need(args.logdir)

agent_algorithm = create_algorithm(exp_config.get_algo_config(0))

# observation_shapes = [(hparams["env"]["obs_size"],)]
# history_len = hparams["server"]["history_length"]
# state_shapes = [(history_len, hparams["env"]["obs_size"],)]
# action_size = hparams["env"]["action_size"]
#
# ############################# define algorithm ############################
# agent_algorithm = create_algorithm(args.agent, hparams)
#
# ############################## run rl server ##############################
# rl_server = RLServer(
#     action_size=action_size,
#     observation_shapes=observation_shapes,
#     state_shapes=state_shapes,
#     agent_algorithm=agent_algorithm,
#     ckpt_path=args.logdir,
#     **hparams["server"])
# rl_server.start()
