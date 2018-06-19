#!/usr/bin/env python

import gym
import numpy as np

class Pendulum:

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.observation_shapes = [(3,)]
        self.action_size = 1

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        elf.init_action = np.round(np.random.uniform(-2.0, 2.0, size=self.action_size))
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action) 
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info

    def get_total_reward(self):
        return self.total_reward
    
    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(np.random.uniform(-2.0, 2.0, size=self.action_size))
        return self.init_action
