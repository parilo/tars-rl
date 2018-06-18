#!/usr/bin/env python

import argparse
import random
import numpy as np
from osim.env import L2RunEnv
from rl_server.server.rl_client import RLClient
from replay_buffer import AgentBuffer

parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller'
)
parser.add_argument(
    '--visualize', dest='visualize', action='store_true', default=False
)
parser.add_argument(
    '--random_start', dest='random_start', action='store_true', default=False
)
parser.add_argument(
    '--id', dest='id', type=int, default=0
)
args = parser.parse_args()
vis = args.visualize
random_start = args.random_start
id_ = args.id

num_actions = 18
history_len = 3

class ExtRunEnv(L2RunEnv):

    def __init__(self, *args, **kwargs):
        super(L2RunEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, done, info = super(L2RunEnv, self).step(action)
        self.total_reward += reward
        if vis:
            print('{} {}'.format(reward, self.total_reward))
        observation = self.preprocess_obs(observation)
        return observation, reward, done, info

    def preprocess_obs(self, obs):

        # position of the pelvis (rotation, x, y)
        # 0 1 2
        # velocity of the pelvis (rotation, x, y)
        # 3 4 5
        # rotation of each ankle, knee and hip (6 values)
        # 6 7 8 9 10 11
        # angular velocity of each ankle, knee and hip (6 values)
        # 12 13 14 15 16 17
        # position of the center of mass (2 values)
        # 18 19
        # velocity of the center of mass (2 values)
        # 20 21
        # positions (x, y) of head, pelvis, torso, left and right toes,
        # left and right talus (14 values)
        # 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        # strength of left and right psoas: 1 for difficulty < 2,
        # otherwise a random normal variable with mean 1 and standard
        # deviation 0.1 fixed for the entire simulation
        # 36 37
        # next obstacle: x distance from the pelvis,
        # y position of the center relative to the the ground, radius.
        # 38 39 40

        obs = np.array(obs)
        x = obs[1]
        y = obs[2]
        a = obs[0]

        obs[1] = 0

        obs[6] -= a
        obs[7] -= a
        obs[8] -= a
        obs[9] -= a
        obs[10] -= a
        obs[11] -= a

        obs[18] -= x

        obs[22] -= x
        obs[23] -= y
        obs[24] = 0
        obs[25] = 0
        obs[26] -= x
        obs[27] -= y

        obs[28] -= x
        obs[29] -= y
        obs[30] -= x
        obs[31] -= y
        obs[32] -= x
        obs[33] -= y
        obs[34] -= x
        obs[35] -= y

        obs[38] /= 100.0
        return obs.tolist()

    def reset(self, *args, **kwargs):
        self.total_reward = 0
        return self.preprocess_obs(super(L2RunEnv, self).reset(*args, **kwargs))

    def get_total_reward(self):
        return self.total_reward


env = ExtRunEnv(visualize=vis)
rl_client = RLClient(port=8777 + id_)

prev_observation = np.array(env.reset())
agent_buffer = AgentBuffer(1010, history_len=history_len)
agent_buffer.push_init_observation([prev_observation])

class InitActionProducer(object):
    def __init__(self):
        self._init_action = None
    def reset_init_action(self):
        self._init_action = np.round(np.random.uniform(0, 0.7, size=18))
    def get_init_action(self):
        return self._init_action

episode_index = 0
step_index = 0
init_action_producer = InitActionProducer()
init_action_producer.reset_init_action()

np.set_printoptions(suppress=True)

def prep_state(state):
    state_ = np.copy(state)
    state_[0] = state_[1] - state_[0]
    state_[1] = state_[2] - state_[1]
    return state_.ravel()

while True:
    
    state = prep_state(agent_buffer.get_current_state()[0])
    
    # randomize
    if (step_index < 20 and random_start):
        if step_index == 10:
            init_action_producer.reset_init_action()
        action = init_action_producer.get_init_action()
    else:
        action_received = rl_client.act([state])
        action = (np.array(action_received) + np.random.normal(scale=0.02, size=num_actions))
        action = np.clip(action, 0.0, 1.0)

    next_observation, reward, done, info = env.step(action.tolist())
    next_observation = np.array(next_observation)
    
    transition = [[next_observation], action, reward, done]
    agent_buffer.push_transition(transition) 
    next_state = prep_state(agent_buffer.get_current_state()[0])

    prev_observation = next_observation
    step_index += 1

    if done:
        
        episode = agent_buffer.get_complete_episode()
        rl_client.store_exp_batch(episode)

        print('--- episode ended {} {} {}'.format(episode_index, step_index, env.get_total_reward()))

        with open('episode_rewards.txt', 'a') as f:
            f.write(str(episode_index) + ' ' + str(env.get_total_reward()) + '\n')

        step_index = 0
        episode_index += 1
        init_action_producer.reset_init_action()
        rand = random.uniform(0, 1)
        prev_observation = np.array(env.reset())
        
        agent_buffer = AgentBuffer(1010, history_len=history_len)
        agent_buffer.push_init_observation([prev_observation])