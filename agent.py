#!/usr/bin/env python

import argparse
import random
import numpy as np
import gym
from osim.env import L2RunEnv
from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from envs.l2run import ExtRunEnv
from envs.pendulum import Pendulum
from envs.lunar_lander import LunarLander

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
parser.add_argument('--frame_skip',
                    dest='frame_skip',
                    default=1)
args = parser.parse_args()

experiment_name = "lunar_lander-hist_len3-frame_skip1-relu-batchnorm-agents40"

#env = Pendulum(frame_skip=args.frame_skip)
#env = ExtRunEnv(frame_skip=args.frame_skip)
env = LunarLander(frame_skip=args.frame_skip)
observation_shapes = env.observation_shapes
action_size = env.action_size

buf_capacity = 1001
history_len = 3

rl_client = RLClient(port=8777+args.id)
agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)

obs = env.reset()
agent_buffer.push_init_observation([obs])

episode_index = 0

while True:
    
    state = agent_buffer.get_current_state(history_len=history_len)[0].ravel()
    
    if (env.time_step < (20 / args.frame_skip) and args.random_start):
        if env.time_step == (10 / args.frame_skip):
            action = env.get_random_action(resample=True)
        else:
            action = env.get_random_action(resample=False)
    else:
        action_received = rl_client.act([state])
        action = np.array(action_received) + np.random.normal(scale=0.02, size=action_size)
        #action = np.clip(action, 0.0, 1.0)
        #action = np.clip(action, 0.0, 1.0)*4-2
        action = np.clip(action, 0.0, 1.0)*2-1

    next_obs, reward, done, info = env.step(action)
    
    transition = [[next_obs], action, reward, done]
    agent_buffer.push_transition(transition) 
    next_state = agent_buffer.get_current_state(history_len=history_len)[0].ravel()

    if done:
        
        episode = agent_buffer.get_complete_episode()
        rl_client.store_episode(episode)

        print('--- episode ended {} {} {}'.format(episode_index, env.time_step, env.get_total_reward()))

        with open('results/' + experiment_name + '_episode_rewards.txt', 'a') as f:
            f.write(str(args.id) + ' ' + str(episode_index) + ' ' + str(env.get_total_reward()) + '\n')

        episode_index += 1
        obs = env.reset()
        agent_buffer = AgentBuffer(buf_capacity, observation_shapes, action_size)
        agent_buffer.push_init_observation([obs])