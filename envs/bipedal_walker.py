#!/usr/bin/env python

import gym
import numpy as np

class BipedalWalker:

    def __init__(self, frame_skip=1, visualize=False):
        self.visualize = visualize
        self.env = gym.make("BipedalWalkerHardcore-v2")
        self.frame_skip = frame_skip
        self.observation_shapes = [(24,)]
        self.action_size = 4

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(-1.0, 1.0, size=self.action_size))
        return self.env.reset()

    def step(self, action):
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            reward = r * 0.1
            # reward for velocity over x
            # reward = observation[2] * 0.1

            if self.visualize:
                self.env.render()

            if done: break
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info

    def get_total_reward(self):
        return self.total_reward

    def get_random_action(self, resample=True):
        if self.time_step % 10 == 0:
            self.init_action = np.round(np.random.uniform(-1.0, 1.0, size=self.action_size))
        return self.init_action
