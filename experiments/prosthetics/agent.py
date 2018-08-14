#!/usr/bin/env python

import sys
sys.path.append("../../")

from misc.defaults import parse_agent_args
from rl_agent import RLAgent

args, hparams = parse_agent_args()

from envs.prosthetics import ProstheticsEnvWrap
env = ProstheticsEnvWrap(
    frame_skip=hparams["env"]["frame_skip"],
    visualize=args.visualize,
    reward_scale=0.1,
    crossing_legs_penalty=10.,
    bending_knees_bonus=1.,
    side_step_penalty=True,
    legs_interleave_bonus=0., #10.,
    # max_episode_length=298
    max_episode_length=1000
)

agent = RLAgent(
    env=env,
    seed=1 + args.id
)
agent.run()
