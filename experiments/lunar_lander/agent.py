#!/usr/bin/env python

import sys
sys.path.append('../../')

import os
import argparse
import random
import numpy as np
import gym

from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from envs.lunar_lander import LunarLander
from misc.experiment_config import ExperimentConfig

# parse input arguments
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--random_start',
                    dest='random_start',
                    action='store_true',
                    default=False)
parser.add_argument('--id',
                    dest='id',
                    type=int,
                    default=0)
parser.add_argument('--visualize',
                    dest='visualize',
                    action='store_true',
                    default=False)
parser.add_argument('--validation',
                    dest='validation',
                    action='store_true',
                    default=False)
parser.add_argument('--experiment_name',
                    dest='experiment_name',
                    type=str,
                    default='experiment')
args = parser.parse_args()

############################## Specify environment and experiment ##############################

C = ExperimentConfig(env_name='lunar_lander', experiment_name=args.experiment_name)

env = LunarLander(frame_skip=C.frame_skip, visualize=args.visualize)
observation_shapes = env.observation_shapes
action_size = env.action_size
if os.path.isfile(C.path_to_rewards_train):
    os.remove(C.path_to_rewards_train)
if os.path.isfile(C.path_to_rewards_test):
    os.remove(C.path_to_rewards_test)

########################################## Train agent #########################################

buf_capacity = 1010

rl_client = RLClient(port=C.port+args.id)
agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
agent_buffer.push_init_observation([env.reset()])

episode_index = 0

while True:

    state = agent_buffer.get_current_state(history_len=C.history_len)[0].ravel()
    
    if args.validation:
        action = rl_client.act([state])
        #action = rl_client.act([state], mode='sac_deterministic')
    else:
        action_received = rl_client.act([state])
        action = np.array(action_received) + np.random.normal(scale=0.02, size=action_size)
    action = np.clip(action, -1., 1.)

    next_obs, reward, done, info = env.step(action)
    transition = [[next_obs], action, reward, done]
    agent_buffer.push_transition(transition)
    next_state = agent_buffer.get_current_state(history_len=C.history_len)[0].ravel()

    if done:
        episode = agent_buffer.get_complete_episode()
        rl_client.store_episode(episode)
        print('--- episode ended {} {} {}'.format(episode_index, env.time_step, env.get_total_reward()))

        if args.validation:
            path_to_rewards = C.path_to_rewards_test
        else:
            path_to_rewards = C.path_to_rewards_train
        with open(path_to_rewards, 'a') as f:
            f.write(str(args.id) + ' ' + str(episode_index) + ' ' + str(env.get_total_reward()) + '\n')

        episode_index += 1
        agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
        agent_buffer.push_init_observation([env.reset()])
