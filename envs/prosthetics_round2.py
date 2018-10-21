import math
import copy
import random
import numpy as np
from osim.env import ProstheticsEnv, rect
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
            activations_penalty=0.,
            num_of_augmented_targets=4,
            bonus_for_knee_angles_scale=0.,
            bonus_for_knee_angles_angle=0.):

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
        self.num_of_augmented_targets = num_of_augmented_targets
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

    def generate_new_augmented_targets(self, poisson_lambda = 50):
        nsteps = self.max_ep_length + 3
        rg = np.array(range(nsteps))
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        velocity[0] = random.uniform(-0.5,0.5)
        heading[0] = random.uniform(-math.pi/8,math.pi/8)

        change = np.cumsum(np.random.poisson(poisson_lambda, 10))

        for i in range(1,nsteps):
            velocity[i] = velocity[i-1]
            heading[i] = heading[i-1]

            if i in change:
                velocity[i] += random.uniform(-0.5,0.5)
                heading[i] += random.uniform(-math.pi/8,math.pi/8)

        trajectory_polar = np.vstack((velocity,heading)).transpose()
        return np.apply_along_axis(rect, 1, trajectory_polar)

    def augmented_reward_round2(self, target):
        state_desc = self.env.get_state_desc()
        prev_state_desc = self.env.get_prev_state_desc()
        penalty = 0

        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.env.osim_model.get_activations())**2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        # augmented_target = self.augmented_targets[self.time_step,:]
        penalty += (state_desc["body_vel"]["pelvis"][0] - target[0])**2
        penalty += (state_desc["body_vel"]["pelvis"][2] - target[2])**2
        
        # Reward for not falling
        reward = 10.0
        
        return reward - penalty 

    def reset(self):
        self.time_step = 0
        self.init_action = np.round(
            np.random.uniform(0, 0.7, size=self.action_size))
        self.total_reward = 0.
        self.total_reward_shaped = 0.
        
        self.augmented_targets = [self.generate_new_augmented_targets() for _ in range(self.num_of_augmented_targets)]

        state_desc = self.env.reset(project=False)
        if self.randomized_start:
            state = get_simbody_state(state_desc)
            # noise = np.random.normal(scale=0.4, size=72)
            # noise[9] = 1.  # left leg
            # noise[6] = 1.  # right leg
            # state = (np.array(state) + noise).tolist()
            
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
        augmented_targets_at_step = [[] for _ in range(self.frame_skip)]
        augmented_targets_rewards = [0 for _ in range(self.num_of_augmented_targets)]
        augmented_targets_observations = [[] for _ in range(self.frame_skip)]
        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            reward_origin += r
            for ati in range(self.num_of_augmented_targets):
                augmented_target = self.augmented_targets[ati][self.time_step].tolist()
                augmented_targets_at_step[i].append(augmented_target)
                augmented_targets_rewards[ati] += self.shape_reward(
                    self.augmented_reward_round2(augmented_target)
                ) * self.reward_scale
                augmented_observation = copy.deepcopy(observation)
                augmented_observation["target_vel"] = augmented_target
                augmented_observation = preprocess_obs_round2(augmented_observation, self.time_step)
                augmented_targets_observations[i].append(augmented_observation)
            done = self.is_done(observation)
            reward += self.shape_reward(r)
            if done:
                self.episodes = self.episodes + 1
                break
    
        observation = preprocess_obs_round2(observation, self.time_step)
        reward *= self.reward_scale
        info["reward_origin"] = reward_origin
        info["augmented_targets"] = {
            'targets': augmented_targets_at_step,
            'rewards': augmented_targets_rewards,
            'observations': augmented_targets_observations
        }
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
        # print('--- knee {} {}'.format(
        #     self.bonus_for_knee_angles_scale * math.exp(-((r_knee_flexion + self.bonus_for_knee_angles_angle) * 6.0)**2),
        #     self.bonus_for_knee_angles_scale * math.exp(-((l_knee_flexion + self.bonus_for_knee_angles_angle) * 6.0)**2)
        # ))
        
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
