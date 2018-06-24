#!/usr/bin/env python

from osim.env import ProstheticsEnv
import numpy as np


class ProstheticsEnvWrap:

    def __init__(self, frame_skip=1, visualize=False):
        self.env = ProstheticsEnv(visualize=visualize)
        self.env.change_model(model='3D', prosthetic=True, difficulty=0, seed=25)
        self.frame_skip = frame_skip
        self.observation_shapes = [(158,)]
        self.action_size = 19

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(0, 1.0, size=self.action_size))
        obs = self.env.reset(project=True)
        return self.preprocess_obs(obs)

    def step(self, action):
        reward = 0
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action, project=True)
            reward += r*0.1
            if done: break
        observation = self.preprocess_obs(observation)
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info

    def preprocess_obs(self, obs):

        obs = np.array(obs)
        obs[[34, 52, 64, 67, 77, 86, 87, 91, 94]] /= 1e3
        obs[[3, 7, 12, 13, 16, 21, 25, 30, 31, 48, 49, 78]] /= 1e2
        obs[[4, 6, 14, 15, 23, 24, 33, 51, 90, 156]] /= 1e1
        return obs

    def get_total_reward(self):
        return self.total_reward

    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(np.random.uniform(0, 1.0, size=self.action_size))
        return self.init_action
