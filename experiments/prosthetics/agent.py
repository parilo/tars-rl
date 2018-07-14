#!/usr/bin/env python

import sys
sys.path.append('../../')

import os
import json
import argparse
import random
import numpy as np
import gym
from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from envs.prosthetics_new import ProstheticsEnvWrap

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
args = parser.parse_args()

############################## Specify environment and experiment ##############################

environment_name = 'prosthetics_new'
experiment_config = json.load(open('config.txt'))

history_len = experiment_config['history_len']
frame_skip = experiment_config['frame_skip']

experiment_file = 'cat-ge'
for i in ['history_len', 'frame_skip', 'n_step', 'batch_size']:
    experiment_file = experiment_file + '-' + i + str(experiment_config[i])
if experiment_config['prio']:
    experiment_file = experiment_file + '-prio'
if experiment_config['sync']:
    experiment_file = experiment_file + '-sync'
else:
    experiment_file = experiment_file + '-async'
path_to_results = 'results/' + experiment_file + '-rewards.txt'
path_to_results_validation = 'results/' + experiment_file + '-rewards-validation.txt'

#if os.path.isfile(path_to_results):
#    os.remove(path_to_results)

port = experiment_config['port']

env = ProstheticsEnvWrap(frame_skip=frame_skip, visualize=args.visualize)
observation_shapes = env.observation_shapes
action_size = env.action_size

########################################## Train agent #########################################

buf_capacity = 1010

rl_client = RLClient(port=port+args.id)
agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)

agent_buffer.push_init_observation([env.reset()])

episode_index = 0

# exploration parameters for gradient exploration
explore_start_temp = 0.02
explore_end_temp = 0.02
explore_episodes = 500
explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
explore_temp = explore_start_temp

expl_sigma = 2e-2*(args.id % 4)

while True:

    state = agent_buffer.get_current_state(history_len=history_len)[0].ravel()

    if (env.time_step < (20 / frame_skip) and args.random_start):
        if env.time_step == (10 // frame_skip):
            action = env.get_random_action(resample=True)
        else:
            action = env.get_random_action(resample=False)
    else:
        # normal noise exploration
        action_received = rl_client.act([state])
        action = np.array(action_received) + np.random.normal(scale=expl_sigma, size=action_size)
        action = np.clip(action, 0., 1.)

        # gradient exploration
        # result = rl_client.act_with_gradient_batch([state])
        # action_received = result[0][0]
        # grad = result[1][0]
        # action = np.array(action_received)
        # if not args.validation:
        #     random_action = env.get_random_action()
        #     explore = 1. - np.clip(np.abs(grad), 0., explore_temp) / explore_temp
        #     action = np.multiply(action, (1. - explore)) + np.multiply(random_action, explore)

        action = np.clip(action, 0., 1.)

    next_obs, reward, done, info = env.step(action)
    transition = [[next_obs], action, reward, done]
    agent_buffer.push_transition(transition)
    next_state = agent_buffer.get_current_state(history_len=history_len)[0].ravel()

    if done:
        episode = agent_buffer.get_complete_episode()
        rl_client.store_episode(episode)
        print('--- episode ended {} {} {}'.format(episode_index, env.time_step, env.get_total_reward()))
        with open(path_to_results_validation if args.validation else path_to_results, 'a') as f:
            f.write(str(args.id) + ' ' + str(episode_index) + ' ' + str(env.get_total_reward()) + '\n')
        episode_index += 1
        agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
        agent_buffer.push_init_observation([env.reset()])
