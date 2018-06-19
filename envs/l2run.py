#!/usr/bin/env python

from osim.env import L2RunEnv
import numpy as np


class ExtRunEnv(L2RunEnv):

    def __init__(self, frame_skip=1):
        self.env = L2RunEnv(visualize=False)
        self.frame_skip = frame_skip
        self.observation_shapes = [(41,)]
        self.action_size = 18

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(0, 0.7, size=self.action_size))
        obs = self.env.reset()
        return self.preprocess_obs(obs)

    def step(self, action):
        reward = 0
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action) 
            reward += r
            if done: break
        observation = self.preprocess_obs(observation)
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info

    def preprocess_obs(self, obs):

        # 0-2   -- position of the pelvis (rotation, x, y)
        # 3-5   -- velocity of the pelvis (rotation, x, y)
        # 6-11  -- rotation of each ankle, knee and hip (6 values)
        # 12-17 -- angular velocity of each ankle, knee and hip (6 values)
        # 18-19 -- position of the center of mass (x, y)
        # 20-21 -- velocity of the center of mass (x, y)
        # 22-35 -- positions (x, y) of head, pelvis, torso, left and 
        #          right toes, left and right talus (14 values)
        # 36-37 -- strength of left and right psoas: 1 for difficulty < 2,
        #          otherwise a random normal variable with mean 1 and standard
        #          deviation 0.1 fixed for the entire simulation
        # 38-40 -- next obstacle: x distance from the pelvis,
        #          y position of the center relative to the the ground, radius

        obs = np.array(obs)
        x = obs[1]
        y = obs[2]
        a = obs[0]
        obs[[1, 24, 25]] = 0
        obs[6:12] += a
        obs[[18, 22, 26, 28, 30, 32, 34]] -= x
        obs[[19, 23, 27, 29, 31, 33, 35]] -= y
        obs[38] /= 100.0
        return obs

    def get_total_reward(self):
        return self.total_reward
    
    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(np.random.uniform(0, 0.7, size=self.action_size))
        return self.init_action