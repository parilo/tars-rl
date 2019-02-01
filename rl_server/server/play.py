#!/usr/bin/env python

from misc.common import parse_play_args
from misc.config import load_config
from rl_server.server.agent import run_agent
from rl_server.server.run_agents import get_algo_and_agent_config

args = parse_play_args()
config = load_config(args.config)

algo_config, agent_config = get_algo_and_agent_config(
    config,
    args.algorithm_id,
    args.agent_id,
    args.seed
)

run_agent(
    config,
    agent_config,
    checkpoint_path=args.checkpoint
)
