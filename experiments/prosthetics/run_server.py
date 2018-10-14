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
algo_config = config.algo_configs[0]

agent_algorithm = create_algorithm(
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    action_size=action_size,
    algo_config=algo_config,
    scope_postfix="algo"
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
