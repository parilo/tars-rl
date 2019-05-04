#!/usr/bin/env python

from rl_server.tensorflow.obs_server import ObsServer
from misc.common import create_if_need, set_global_seeds, parse_server_args
from misc.config import load_config


args = parse_server_args()
exp_config = load_config(args.config)
server = ObsServer(exp_config)
server.start()
