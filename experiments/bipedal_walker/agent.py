#!/usr/bin/env python

import sys
sys.path.append("../../")

from misc.defaults import parse_agent_args
from rl_agent import RLAgent

args, hparams = parse_agent_args()

from envs.bipedal_walker import BipedalWalker
env = BipedalWalker(
    frame_skip=hparams["env"]["frame_skip"],
    visualize=args.visualize
)

agent = RLAgent(
    env=env,
    seed=1
)
agent.run()
