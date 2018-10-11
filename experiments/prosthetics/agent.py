#!/usr/bin/env python

import sys
sys.path.append("../../")

from misc.defaults import parse_agent_args
from misc.config import EnsembleConfig
from rl_agent import RLAgent

args = parse_agent_args()
experiment_config = EnsembleConfig(args.config)

from envs.prosthetics_round2 import ProstheticsEnvWrap

env = ProstheticsEnvWrap(
    frame_skip=experiment_config.config["env"]["frame_skip"],
    visualize=args.visualize,
    reward_scale=experiment_config.config["env"]["reward_scale"],
    crossing_legs_penalty=0.,
    bending_knees_bonus=0.,
    left_knee_bonus=0.,
    right_knee_bonus=0.,
    activations_penalty=experiment_config.config["env"]["activations_penalty"],
    max_reward=experiment_config.config["env"]["max_reward"],
    action_func=experiment_config.config["env"]["action_func"],
    max_episode_length=1000
)

agent = RLAgent(
    experiment_config,
    experiment_config.algo_configs[args.id],
    logdir=args.logdir,
    validation=args.validation,
    exploration=args.exploration,
    store_episodes=args.store_episodes,
    agent_id=args.id,
    env=env,
    seed=1 + args.id
)
agent.run()
