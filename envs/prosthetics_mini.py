import os
from osim.env import ProstheticsEnv
import numpy as np


class ProstheticsMini:

    def __init__(
            self,
            frame_skip=1,
            visualize=False,
            randomized_start=False,
            max_episode_length=300,
            reward_scale=0.1,
            death_penalty=0.0,
            living_bonus=0.0,
            side_deviation_penalty=0.0,
            crossing_legs_penalty=0.0,
            bending_knees_bonus=0.0,
            side_step_penalty=False,
            legs_interleave_bonus=0.):

        self.vis = visualize
        self.randomized_start = randomized_start
        self.env = ProstheticsEnv(
            visualize=visualize, integrator_accuracy=1e-3)
        self.env.change_model(
            model="3D", prosthetic=True, difficulty=0)
        self.frame_skip = frame_skip
        self.observation_shapes = [(72,)]
        self.action_size = 19
        self.max_ep_length = max_episode_length

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.side_dev_coef = side_deviation_penalty
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus
        self.side_step_penalty = side_step_penalty
        self.legs_interleave_bonus = legs_interleave_bonus
        self.front_leg = 0
        self.prev_legs_x = np.array([0., 0.])

        self.obs_list = []

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.total_reward_shaped = 0.
        self.init_action = np.round(
            np.random.uniform(0, 0.7, size=self.action_size))
        obs = self.env.reset(project=False)
        prep_obs = self.preprocess_obs(obs)

        if self.randomized_start:
            noise = np.random.normal(scale=0.1, size=72)
            noise[4] = 0
            prep_obs = (np.array(prep_obs) + noise).tolist()
            self.set_state(prep_obs)

        return prep_obs

    def step(self, action):

        reward = 0
        action = np.clip(action, 0.0, 1.0)
        for i in range(self.frame_skip):
            observation, r, _, info = self.env.step(action, project=False)
            self.original_reward = r
            self.total_reward += r
            done = self.is_done(observation)
            reward += self.shape_reward(r) * self.reward_scale
            self.total_reward_shaped += reward
            if done:
                break

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
        vy, vz = state_desc["body_vel"]["pelvis"][1:]
        side_dev_penalty = (vy ** 2 + vz ** 2)
        reward -= self.side_dev_coef * side_dev_penalty

        # crossing legs penalty
        pelvis_xy = np.array(state_desc["body_pos"]["pelvis"])
        left = np.array(state_desc["body_pos"]["toes_l"]) - pelvis_xy
        right = np.array(state_desc["body_pos"]["pros_foot_r"]) - pelvis_xy
        axis = np.array(state_desc["body_pos"]["head"]) - pelvis_xy
        cross_legs_penalty = np.cross(left, right).dot(axis)
        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0
        reward += self.cross_legs_coef * cross_legs_penalty

        # bending knees bonus
        r_knee_flexion = np.minimum(state_desc["joint_pos"]["knee_r"][0], 0.)
        l_knee_flexion = np.minimum(state_desc["joint_pos"]["knee_l"][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_coef * bend_knees_bonus

        # legs interleaving
        pelvis_x = pelvis_xy[0]
        legs_x = np.array([
            state_desc["body_pos"]["pros_tibia_r"][0] - pelvis_x,
            state_desc["body_pos"]["tibia_l"][0] - pelvis_x])
        if legs_x[self.front_leg] > 0.225:
            # change leg
            self.front_leg = 1 - self.front_leg
        legs_v = legs_x - self.prev_legs_x
        reward_for_front_leg = 0.
        front_leg_v = legs_v[self.front_leg]
        if front_leg_v > 0:
            reward_for_front_leg = self.legs_interleave_bonus * front_leg_v
        self.prev_legs_x = legs_x
        reward += reward_for_front_leg

        # side step penalty
        if self.side_step_penalty:
            rx, ry, rz = state_desc["body_pos_rot"]["pelvis"]
            R = euler_angles_to_rotation_matrix([rx, ry, rz])
            # reward = 0.5 * reward * (1.0 + (1.0 - math.fabs(R[2, 0])))
            reward = reward * (1.0 - math.fabs(R[2, 0]))

        return reward

    def preprocess_obs(self, obs):

        res = []
        # joints
        for joint_val in ["joint_pos", "joint_vel"]:
            for joint in ["ground_pelvis", "hip_r", "hip_l", "back",
                          "knee_r", "knee_l", "ankle_r", "ankle_l"]:
                res += obs[joint_val][joint]

        # muscles
        muscles = ["abd_r", "add_r", "hamstrings_r", "bifemsh_r",
                   "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
                   "abd_l", "add_l", "hamstrings_l", "bifemsh_l",
                   "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
                   "gastroc_l", "soleus_l", "tib_ant_l"]
        for muscle in muscles:
            res += [obs["muscles"][muscle]["activation"]]
            res += [obs["muscles"][muscle]["fiber_length"]]
        return res

    def set_state(self, obs):
        simbody_state = self.env.osim_model.get_state()
        obj = simbody_state.getY()
        for i in range(72):
            obj[i] = obs[i]
        self.env.osim_model.set_state(simbody_state)

    def get_total_reward(self):
        return self.total_reward

    def get_total_reward_shaped(self):
        return self.total_reward_shaped

    def get_random_action(self, resample=True):
        if resample:
            self.init_action = np.round(
                np.random.uniform(0, 0.7, size=self.action_size))
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
