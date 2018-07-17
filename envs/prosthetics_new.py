#!/usr/bin/env python

from osim.env import ProstheticsEnv
import numpy as np


class ProstheticsEnvWrap:

    def __init__(self, frame_skip=1,
                 visualize=False,
                 max_episode_length=300,
                 reward_scale=0.1,
                 death_penalty=0.0,
                 living_bonus=0.0,
                 side_deviation_penalty=0.0,
                 crossing_legs_penalty=0.0,
                 bending_knees_bonus=0.0):

        self.env = ProstheticsEnv(visualize=visualize, integrator_accuracy=1e-3)
        self.env.change_model(model='3D', prosthetic=True, difficulty=0, seed=np.random.randint(200))
        self.frame_skip = frame_skip
        self.observation_shapes = [(333,)]
        self.action_size = 19
        self.max_ep_length = max_episode_length

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.side_dev_coef = side_deviation_penalty
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(0, 0.7, size=self.action_size))
        obs = self.env.reset(project=False)
        return self.preprocess_obs(obs)

    def step(self, action):

        reward = 0
        action = np.clip(action, 0.0, 1.0)
        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            self.total_reward += r
            done = self.is_done(observation)
            reward += self.shape_reward(r) * self.reward_scale
            if done: break

        observation = self.preprocess_obs(observation)
        self.time_step += 1
        return observation, reward, done, info

    def is_done(self, observation):
        pelvis_y = observation["body_pos"]["pelvis"][1]
        if self.time_step * self.frame_skip > self.max_ep_length:
            return True
        elif pelvis_y < 0.6:
            return True
        return False

    def shape_reward(self, reward):
        state_desc = self.env.get_state_desc()

        # death penalty
        if self.time_step * self.frame_skip < self.max_ep_length:
            reward -= self.death_penalty
        else:
            reward += self.living_bonus

        # deviation from forward direction penalty
        vy, vz = state_desc['body_vel']['pelvis'][1:]
        side_dev_penalty = (vy ** 2 + vz ** 2)
        reward -= self.side_dev_coef * side_dev_penalty

        # crossing legs penalty
        pelvis_xy = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xy
        right = np.array(state_desc['body_pos']['pros_foot_r']) - pelvis_xy
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xy
        cross_legs_penalty = np.sign(np.cross(left, right).dot(axis))
        reward -= self.cross_legs_coef * cross_legs_penalty

        # bending knees bonus
        r_knee_flexion = np.min(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.min(state_desc['joint_pos']['knee_l'][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_coef * bend_knees_bonus

        return reward

    def preprocess_obs(self, obs):

        res = []

        force_mult = 5e-4
        acc_mult = 1e-2

        # body parts
        body_parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                      'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
        # calculate linear coordinates of pelvis to switch
        # to the reference frame attached to the pelvis
        x, y, z = obs['body_pos']['pelvis']
        # 30 components -- relative linear coordinates of body parts (except pelvis)
        for body_part in body_parts:
            if body_part != 'pelvis':
                x_, y_, z_ = obs['body_pos'][body_part]
                res += [x_-x, y_-y, z_-z]
        # 2 components -- relative linear coordinates of center mass
        x_, y_ = obs['misc']['mass_center_pos']
        res += [x_-x, y_-y]
        # 35 components -- linear velocities of body parts (and center mass)
        for body_part in body_parts:
            res += obs['body_vel'][body_part]
        res += obs['misc']['mass_center_vel']
        # 35 components -- linear accelerations of body parts (and center mass)
        for body_part in body_parts:
            res += obs['body_acc'][body_part]
        res += obs['misc']['mass_center_acc']
        # calculate angular coordinates of pelvis to switch
        # to the reference frame attached to the pelvis
        rx, ry, rz = obs['body_pos_rot']['pelvis']   
        # 30 components -- relative angular coordinates of body parts (except pelvis)
        for body_part in body_parts:
            if body_part != 'pelvis':
                rx_, ry_, rz_ = obs['body_pos_rot'][body_part]
                res += [rx_-rx, ry_-ry, rz_-rz]
        # 33 components -- linear velocities of body parts
        for body_part in body_parts:
            res += obs['body_vel_rot'][body_part]
        # 33 components -- linear accelerations of body parts
        for body_part in body_parts:
            res += obs['body_acc_rot'][body_part]
        
        # joints
        for joint_val in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r',
                          'hip_l', 'knee_l', 'ankle_l']:
                res += obs[joint_val][joint][:3]

        # muscles
        muscles = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 
           'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 
           'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 
           'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 
           'gastroc_l', 'soleus_l', 'tib_ant_l']
        for muscle in muscles:
            res += [obs['muscles'][muscle]['activation']]
            res += [obs['muscles'][muscle]['fiber_length']]
            res += [obs['muscles'][muscle]['fiber_velocity']]
        for muscle in muscles:
            res += [obs['muscles'][muscle]['fiber_force']*force_mult]
        forces = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r',
          'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 
          'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 
          'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 
          'gastroc_l', 'soleus_l', 'tib_ant_l', 'ankleSpring', 
          'pros_foot_r_0', 'foot_l', 'HipLimit_r', 'HipLimit_l', 
          'KneeLimit_r', 'KneeLimit_l', 'AnkleLimit_r', 'AnkleLimit_l', 
          'HipAddLimit_r', 'HipAddLimit_l',]

        # forces
        for force in forces:
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
            self.init_action = np.round(np.random.uniform(0, 0.7, size=self.action_size))
        return self.init_action

    def generate_random_actions(self, num_actions=1000, max_repeat=5):
        """generate a number of random actions for initial exploration"""
        actions = []

        while len(actions) < num_actions:
            # choose random action type
            if np.random.rand() < 0.3:
                action = np.random.uniform(0., 1., size=self.action_size)
            else:
                action = self.get_random_action(resample=True)
            # choose how many times action should be repeated
            act_repeat = np.random.randint(max_repeat) + 1
            actions += [action for i in range(act_repeat)]

        return actions[:num_actions]
