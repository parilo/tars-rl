#!/usr/bin/env python

from osim.env import ProstheticsEnv
import numpy as np


class ProstheticsEnvWrap:

    def __init__(self, frame_skip=1, visualize=False):
        self.env = ProstheticsEnv(visualize=visualize)
        self.env.change_model(model='3D', prosthetic=True, difficulty=0, seed=25)
        self.frame_skip = frame_skip
        self.observation_shapes = [(333,)]
        self.action_size = 19

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(0, 1.0, size=self.action_size))
        obs = self.env.reset(project=False)
        return self.preprocess_obs(obs)

    def step(self, action):
        reward = 0
        action = np.clip(action, 0.0, 1.0)
        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            done = self.is_done(observation)
            reward += r*1e-2
            if done: break

        observation = self.preprocess_obs(observation)
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info
    
    def is_done(self, observation):
        pelvis_y = observation["body_pos"]["pelvis"][1]
        if (self.time_step * self.frame_skip > 999):
            return True
        elif pelvis_y < 0.6:
            return True
        return False

    def preprocess_obs(self, obs):

        res = []

        force_mult = 5e-4
        acc_mult = 1e-2

        # calculate linear coordinates of pelvis to switch
        # to the reference frame attached to the pelvis
        x, y, z = obs['body_pos']['pelvis']

        # 30 components -- relative linear coordinates of body parts (except pelvis)
        for body_part in obs['body_pos'].keys():
            if body_part != 'pelvis':
                x_, y_, z_ = obs['body_pos'][body_part]
                res += [x_-x, y_-y, z_-z]
        # 2 components -- relative linear coordinates of center mass
        x_, y_ = obs['misc']['mass_center_pos']
        res += [x_-x, y_-y]
        # 35 components -- linear velocities of body parts (and center mass)
        for body_part in obs['body_vel'].keys():
            res += obs['body_vel'][body_part]
        res += obs['misc']['mass_center_vel']
        # 35 components -- linear accelerations of body parts (and center mass)
        for body_part in obs['body_acc'].keys():
            res += obs['body_acc'][body_part]
        res += obs['misc']['mass_center_acc']

        # calculate angular coordinates of pelvis to switch
        # to the reference frame attached to the pelvis
        rx, ry, rz = obs['body_pos_rot']['pelvis']   
        # 30 components -- relative angular coordinates of body parts (except pelvis)
        for body_part in obs['body_pos_rot'].keys():
            if body_part != 'pelvis':
                rx_, ry_, rz_ = obs['body_pos_rot'][body_part]
                res += [rx_-rx, ry_-ry, rz_-rz]
        # 33 components -- linear velocities of body parts
        for body_part in obs['body_vel_rot'].keys():
            res += obs['body_vel_rot'][body_part]

        # 33 components -- linear accelerations of body parts
        for body_part in obs['body_acc_rot'].keys():
            res += obs['body_acc_rot'][body_part]
        
        # joints
        for joint_val in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r',
                          'hip_l', 'knee_l', 'ankle_l']:
                res += obs[joint_val][joint][:3]

        # muscles
        for muscle in obs['muscles'].keys():
            res += [obs['muscles'][muscle]['activation']]
            res += [obs['muscles'][muscle]['fiber_length']]
            res += [obs['muscles'][muscle]['fiber_velocity']]

        for muscle in obs['muscles'].keys():
            res += [obs['muscles'][muscle]['fiber_force']*force_mult]
        for force in obs['forces'].keys():
            f = obs['forces'][force]
            if len(f) == 1: res+= [f[0]*force_mult]

        res = np.array(res)
        res[67:102] *= acc_mult
        res[165:198] *= acc_mult
        res[224:237] *= acc_mult

        return res.tolist()

    def get_total_reward(self):
        return self.total_reward

    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(np.random.uniform(0, 1.0, size=self.action_size))
        return self.init_action