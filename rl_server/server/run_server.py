#!/usr/bin/env python

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.algo_fabric import create_algorithm
from misc.common import create_if_need, set_global_seeds, parse_server_args
from misc.config import ExperimentConfig


args = parse_server_args()
exp_config = ExperimentConfig(args.config)
set_global_seeds(exp_config.server.seed)
create_if_need(exp_config.server.logdir)

agent_algorithm = create_algorithm(exp_config)

rl_server = RLServer(exp_config, agent_algorithm)
if hasattr(exp_config.server, 'load_checkpoint'):
    rl_server.load_weights(exp_config.server.load_checkpoint)
rl_server.start()
