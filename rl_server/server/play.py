#!/usr/bin/env python

from misc.common import parse_play_args
from misc.config import load_config
from rl_server.server.agent import run_agent

args = parse_play_args()
config = load_config(args.config)
ps = []
agent_id = 0


def set_default(obj, param_name, value):
    if param_name not in obj:
        obj[param_name] = value


exp_config_obj = config.as_obj()
algorithm_id = args.algorithm_id

agent_config = {
    'algorithm_id': algorithm_id,
    'agent_id': agent_id,
    'visualize': True,
    'exploration': None,
    'store_episodes': False,
    'seed': 42
}

run_agent(
    config,
    agent_config,
    checkpoint_path=args.checkpoint
)
