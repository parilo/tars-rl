import math
import numpy as np
from osim.env import ProstheticsEnv
from gym.spaces import Box

from envs.prosthetics_preprocess import preprocess_obs_round2, \
    euler_angles_to_rotation_matrix, get_simbody_state


class ProstheticsEnvWrap:
    def __init__(
            self,
            frame_skip=1,
            visualize=False,
            randomized_start=False,
            max_episode_length=300,
            reward_scale=0.1,
            death_penalty=0.0,
            living_bonus=0.0,
            crossing_legs_penalty=0.0,
            bending_knees_bonus=0.0,
            left_knee_bonus=0.,
            right_knee_bonus=0.,
            max_reward=10.0,
            activations_penalty=0.):

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
            noise = np.random.normal(scale=0.1, size=72)
            noise[3:6] = 0
            state = (np.array(state) + noise).tolist()
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


ENV = ProstheticsEnvWrap
