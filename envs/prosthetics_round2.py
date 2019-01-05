import os
import math
import random

import numpy as np
from osim.env import ProstheticsEnv, rect
from gym.spaces import Box


norm_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'prosthetics_norm_v21.npz')
with np.load(norm_file) as data:
    obs_means = data['means']
    obs_stds = data['stds']


# Calculates Rotation Matrix given euler angles.
def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def preprocess_obs(state_desc):

    res = []

    force_mult = 5e-4
    acc_mult = 1e-2

    # body parts
    body_parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                  'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
    # calculate linear coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    x, y, z = state_desc['body_pos']['pelvis']
    rx, ry, rz = state_desc['body_pos_rot']['pelvis']
    res += [y, z]
    res += [rx, ry, rz]
    # 30 components -- relative linear coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != 'pelvis':
            x_, y_, z_ = state_desc['body_pos'][body_part]
            res += [x_-x, y_-y, z_-z]
    # 2 components -- relative linear coordinates of center mass
    x_, y_, z_ = state_desc['misc']['mass_center_pos']
    res += [x_-x, y_-y, z_-z]
    # 35 components -- linear velocities of body parts (and center mass)
    for body_part in body_parts:
        res += state_desc['body_vel'][body_part]
    res += state_desc['misc']['mass_center_vel']
    # 35 components -- linear accelerations of body parts (and center mass)
    for body_part in body_parts:
        res += state_desc['body_acc'][body_part]
    res += state_desc['misc']['mass_center_acc']
    # calculate angular coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    # 30 components -- relative angular coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != 'pelvis':
            rx_, ry_, rz_ = state_desc['body_pos_rot'][body_part]
            res += [rx_-rx, ry_-ry, rz_-rz]
    # 33 components -- linear velocities of body parts
    for body_part in body_parts:
        res += state_desc['body_vel_rot'][body_part]
    # 33 components -- linear accelerations of body parts
    for body_part in body_parts:
        res += state_desc['body_acc_rot'][body_part]

    # joints
    for joint_val in ['joint_pos', 'joint_vel', 'joint_acc']:
        for joint in ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r',
                      'hip_l', 'knee_l', 'ankle_l']:
            res += state_desc[joint_val][joint][:3]

    # muscles
    muscles = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r',
       'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r',
       'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l',
       'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l',
       'gastroc_l', 'soleus_l', 'tib_ant_l']
    for muscle in muscles:
        res += [state_desc['muscles'][muscle]['activation']]
        res += [state_desc['muscles'][muscle]['fiber_length']]
        res += [state_desc['muscles'][muscle]['fiber_velocity']]
    for muscle in muscles:
        res += [state_desc['muscles'][muscle]['fiber_force']*force_mult]
    forces = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r',
      'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r',
      'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l',
      'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l',
      'gastroc_l', 'soleus_l', 'tib_ant_l', 'ankleSpring',
      'pros_foot_r_0', 'foot_l', 'HipLimit_r', 'HipLimit_l',
      'KneeLimit_r', 'KneeLimit_l', 'AnkleLimit_r', 'AnkleLimit_l',
      'HipAddLimit_r', 'HipAddLimit_l']

    # forces
    for force in forces:
        f = state_desc['forces'][force]
        if len(f) == 1:
            res += [f[0] * force_mult]

    res = (np.array(res) - obs_means) / obs_stds

    return res.tolist()


def get_simbody_state(state_desc):
    res = []
    # joints
    for joint_val in ["joint_pos", "joint_vel"]:
        for joint in ["ground_pelvis", "hip_r", "hip_l", "back",
                      "knee_r", "knee_l", "ankle_r", "ankle_l"]:
            res += state_desc[joint_val][joint]
    # muscles
    muscles = ["abd_r", "add_r", "hamstrings_r", "bifemsh_r",
               "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
               "abd_l", "add_l", "hamstrings_l", "bifemsh_l",
               "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
               "gastroc_l", "soleus_l", "tib_ant_l"]
    for muscle in muscles:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
    return res


def preprocess_obs_round2(state_desc, step_index):
    res = preprocess_obs(state_desc)
    res += state_desc["target_vel"]
    res += [float(step_index) / 500. - 1.]
    return res


class ProstheticsEnvWrap:
    def __init__(
        self,
        reward_scale=0.1,
        frame_skip=1,
        visualize=False,
        reinit_random_action_every=1,
        randomized_start=False,
        max_episode_length=300,
        death_penalty=0.0,
        living_bonus=0.0,
        crossing_legs_penalty=0.0,
        bending_knees_bonus=0.0,
        left_knee_bonus=0.,
        right_knee_bonus=0.,
        max_reward=10.0,
        activations_penalty=0.,
        bonus_for_knee_angles_scale=0.,
        bonus_for_knee_angles_angle=0.
    ):

        self.reinit_random_action_every = reinit_random_action_every
        self.visualize = visualize
        self.randomized_start = randomized_start
        self.env = ProstheticsEnv(visualize=visualize, integrator_accuracy=1e-3)
        self.env.change_model(
            model="3D", prosthetic=True, difficulty=1,
            seed=np.random.randint(200))

        self.frame_skip = frame_skip
        self.observation_shapes = [(345,)]
        self.action_size = 19
        self.max_ep_length = max_episode_length - 2
        self.activations_penalty = activations_penalty
        self.bonus_for_knee_angles_scale = bonus_for_knee_angles_scale
        self.bonus_for_knee_angles_angle = bonus_for_knee_angles_angle

        self.observation_space = Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(344,))
        self.action_space = Box(
            low=self.env.action_space.low[0],
            high=self.env.action_space.high[0],
            shape=(19,))

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus
        self.left_knee_bonus = left_knee_bonus
        self.right_knee_bonus = right_knee_bonus
        self.max_reward = max_reward

        self.episodes = 1
        self.ep2reload = 10

    def reset(self):
        self.time_step = 0
        self.init_action = np.round(
            np.random.uniform(0, 0.7, size=self.action_size))
        self.total_reward = 0.
        self.total_reward_shaped = 0.

        state_desc = self.env.reset(project=False)
        if self.randomized_start:
            state = get_simbody_state(state_desc)

            amplitude = random.gauss(0.8, 0.05)
            direction = random.choice([-1., 1])
            amplitude_knee = random.gauss(-1.2, 0.05)
            state[4] = 0.8
            state[6] = amplitude * direction  # right leg
            state[9] = amplitude * direction * (-1.)  # left leg
            state[13] = amplitude_knee if direction == 1. else 0  # right knee
            state[14] = amplitude_knee if direction == -1. else 0  # left knee
            
            simbody_state = self.env.osim_model.get_state()
            obj = simbody_state.getY()
            for i in range(72):
                obj[i] = state[i]
            self.env.osim_model.set_state(simbody_state)
        return preprocess_obs_round2(state_desc, self.time_step)

    def step(self, action):
        reward = 0
        reward_origin = 0
        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            reward_origin += r
            done = self.is_done(observation)
            reward += self.shape_reward(r)
            if done:
                self.episodes = self.episodes + 1
                break
    
        observation = preprocess_obs_round2(observation, self.time_step)
        reward *= self.reward_scale
        info["reward_origin"] = reward_origin
        self.time_step += 1
        self.total_reward += reward_origin
        self.total_reward_shaped += reward
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

        # crossing legs penalty
        pelvis_xy = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xy
        right = np.array(state_desc['body_pos']['pros_foot_r']) - pelvis_xy
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xy
        cross_legs_penalty = np.cross(left, right).dot(axis)
        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0
        reward += self.cross_legs_coef * cross_legs_penalty

        # bending knees bonus
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_coef * bend_knees_bonus
        
        reward += self.bonus_for_knee_angles_scale * math.exp(-((r_knee_flexion + self.bonus_for_knee_angles_angle) * 6.0)**2)
        reward += self.bonus_for_knee_angles_scale * math.exp(-((l_knee_flexion + self.bonus_for_knee_angles_angle) * 6.0)**2)

        r_knee_flexion = math.fabs(state_desc['joint_vel']['knee_r'][0])
        l_knee_flexion = math.fabs(state_desc['joint_vel']['knee_l'][0])
        reward += r_knee_flexion * self.right_knee_bonus
        reward += l_knee_flexion * self.left_knee_bonus
        
        reward -= np.sum(
            np.array(self.env.osim_model.get_activations())**2
        ) * self.activations_penalty
        
        reward = reward - 10.0 + self.max_reward

        return reward

    def get_total_reward(self):
        return self.total_reward
        
    def get_total_reward_shaped(self):
        return self.total_reward_shaped
