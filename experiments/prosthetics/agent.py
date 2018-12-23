#!/usr/bin/env python

import sys
sys.path.append("../../")

from misc.defaults import parse_agent_args
from misc.config import EnsembleConfig
from rl_server.server.rl_agent_loop import RLAgent

args = parse_agent_args()
experiment_config = EnsembleConfig(args.config)

from envs.prosthetics_round2 import ProstheticsEnvWrap

env = ProstheticsEnvWrap(
    frame_skip=experiment_config.config["env"]["frame_skip"],
    visualize=args.visualize,
    reward_scale=experiment_config.config["env"]["reward_scale"],
    crossing_legs_penalty=experiment_config.config["env"]["crossing_legs_penalty"],
    bending_knees_bonus=0.,
    left_knee_bonus=0.,
    right_knee_bonus=0.,
    activations_penalty=experiment_config.config["env"]["activations_penalty"],
    max_reward=experiment_config.config["env"]["max_reward"],
    max_episode_length=1000,
    num_of_augmented_targets=experiment_config.config["env"]["num_of_augmented_targets"],
    randomized_start=args.random_start,
    bonus_for_knee_angles_scale=experiment_config.config["env"]["bonus_for_knee_angles_scale"],
    bonus_for_knee_angles_angle=experiment_config.config["env"]["bonus_for_knee_angles_angle"]
)

agent = RLAgent(
    experiment_config,
    experiment_config.algo_configs[args.id % experiment_config.get_algos_count()],
    logdir=args.logdir,
    validation=args.validation,
    exploration=args.exploration,
    store_episodes=args.store_episodes,
    agent_id=args.id,
    env=env,
    seed=10 + args.id,
    step_limit=args.step_limit
)
agent.run()
