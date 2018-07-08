#!/usr/bin/env python

from osim.env import L2RunEnv
import numpy as np


class ExtRunEnv(L2RunEnv):

    def __init__(self, frame_skip=1, visualize=False):
        self.vis = visualize
        self.env = L2RunEnv(visualize=visualize)
        self.frame_skip = frame_skip
        self.observation_shapes = [(41,)]
        self.action_size = 18

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(0., 0.7, size=self.action_size))
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

        # res += state_desc["joint_pos"]["ground_pelvis"] 3 rot, x, y
        # res += state_desc["joint_vel"]["ground_pelvis"] 3 rot, x, y
        # 6
        #
        # bedro, koleno, lodizhka
        # for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
        #     res += state_desc["joint_pos"][joint]
        #     res += state_desc["joint_vel"][joint]
        # 2 * 6 = 12 rot, rot_v
        # 18
        #
        # for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        # 2 * 7 = 14 x, y
        # 32
        #
        # res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"]
        # 4
        # 36
        #
        # res += [0]*5
        # 41

        obs = np.array(obs)
        x = obs[1]
        y = obs[2]
        a = obs[0]
        obs[[0, 20, 21]] = 0
        obs[[6, 8, 10, 12, 14, 16]] -= a
        obs[[18, 22, 24, 26, 28, 30, 32]] -= x
        obs[[19, 23, 25, 27, 29, 31, 33]] -= y
        return obs.copy()

    def get_total_reward(self):
        return self.total_reward

    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(np.random.uniform(0., 0.7, size=self.action_size))
        return self.init_action
