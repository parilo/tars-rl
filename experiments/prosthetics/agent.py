#!/usr/bin/env python

import sys
sys.path.append("../../")

import os
import time
import argparse
import numpy as np
import torch

from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from envs.prosthetics_new import ProstheticsEnvWrap
from misc.defaults import default_parse_fn
from tensorboardX import SummaryWriter
from misc.defaults import create_if_need
from datetime import datetime

# parse input arguments
parser = argparse.ArgumentParser(
    description="Train or test neural net motor controller")
parser.add_argument(
    "--random_start",
    dest="random_start",
    action="store_true",
    default=False)
parser.add_argument(
    "--id",
    dest="id",
    type=int,
    default=0)
parser.add_argument(
    "--visualize",
    dest="visualize",
    action="store_true",
    default=False)
parser.add_argument(
    "--validation",
    dest="validation",
    action="store_true",
    default=False)
parser.add_argument(
    "--hparams",
    type=str, required=True)
parser.add_argument(
    "--logdir",
    type=str, required=True)
args = parser.parse_args()

############################## Specify environment and experiment ##############################
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

args, hparams = default_parse_fn(args, [])

env = ProstheticsEnvWrap(
    frame_skip=hparams["env"]["frame_skip"],
    visualize=args.visualize,
    reward_scale=0.1,
    crossing_legs_penalty=10.,
    bending_knees_bonus=1.,
    max_episode_length=1000
)
observation_shapes = env.observation_shapes
action_size = env.action_size
path_to_rewards_train = f"{args.logdir}/rewards-train.txt"
path_to_rewards_test = f"{args.logdir}/rewards-test.txt"

if os.path.isfile(path_to_rewards_train):
    os.remove(path_to_rewards_train)
if os.path.isfile(path_to_rewards_test):
    os.remove(path_to_rewards_test)

if args.validation:
    path_to_rewards = path_to_rewards_test
else:
    path_to_rewards = path_to_rewards_train

current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
if args.validation:
    path_to_rewards = path_to_rewards_test
    logpath = f"{args.logdir}/agent-valid-{args.id}-{current_date}"
else:
    path_to_rewards = path_to_rewards_train
    logpath = f"{args.logdir}/agent-train-{args.id}-{current_date}"
create_if_need(logpath)
logger = SummaryWriter(logpath)

history_len = hparams["server"]["history_length"]

########################################## Train agent #########################################

buf_capacity = 1010

rl_client = RLClient(port=hparams["server"]["init_port"] + args.id)
agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
agent_buffer.push_init_observation([env.reset()])

episode_index = 0
time_step = 0
random_actions = env.generate_random_actions(2000)

# exploration parameters for gradient exploration
explore_start_temp = 0.01
explore_end_temp = 0.01
explore_episodes = 500
explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
explore_temp = explore_start_temp

expl_sigma = 3e-2 * (args.id % 4)

start_time = time.time()
n_steps = 0
episode_index = 0


while True:

    state = agent_buffer.get_current_state(history_len=history_len)[0].ravel()

    if episode_index < 5:
        action = random_actions[time_step]
    else:
        if args.validation:
            action = rl_client.act([state])
        else:
            action = rl_client.act([state])
            action = np.array(action) + np.random.normal(
                scale=expl_sigma, size=action_size)
        action = np.clip(action, -1., 1.)

    next_obs, reward, done, info = env.step(action)
    transition = [[next_obs], action, reward, done]
    agent_buffer.push_transition(transition)
    next_state = agent_buffer.get_current_state(
        history_len=history_len)[0].ravel()
    time_step += 1
    n_steps += 1

    if done:
        elapsed_time = time.time() - start_time
        episode = agent_buffer.get_complete_episode()
        rl_client.store_episode(episode)
        print("--- episode ended {} {} {}".format(
            episode_index, env.time_step, env.get_total_reward()))

        logger.add_scalar("steps", n_steps, episode_index)
        logger.add_scalar(
            "reward", env.get_total_reward(), episode_index)
        logger.add_scalar(
            "episode per minute",
            episode_index / elapsed_time * 60,
            episode_index)
        logger.add_scalar(
            "steps per second",
            n_steps / elapsed_time,
            episode_index)

        with open(path_to_rewards, "a") as f:
            f.write(str(args.id) + " " + str(episode_index) + " " + str(env.get_total_reward()) + "\n")

        episode_index += 1
        agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
        agent_buffer.push_init_observation([env.reset()])
        n_steps = 0
        start_time = time.time()
