import random

import numpy as np
import cv2

class GymEnvWrapper:

    def __init__(
        self,
        env,
        seed=0,
        reward_scale=1.,
        frame_skip=1,
        visualize=False,
        reinit_random_action_every=1,
        max_episode_length=1000,
        obs_is_image=False,
        obs_image_resize_to=None,
        obs_image_to_grayscale=False,
        render_with_cv2=False,
        render_with_cv2_resize=None,
        agent_id=0  # needed to display in cv2 imshow window
    ):
        self.env = env
        self.reward_scale = reward_scale
        self.frame_skip = frame_skip
        self.visualize = visualize
        self.seed = seed
        self.max_episode_length = max_episode_length
        self.obs_is_image = obs_is_image
        self.obs_image_resize_to = obs_image_resize_to
        self.obs_image_to_grayscale = obs_image_to_grayscale
        self.render_with_cv2 = render_with_cv2
        self.render_with_cv2_resize = render_with_cv2_resize
        self.agent_id = agent_id

        self.reinit_random_action_every = reinit_random_action_every
        self.random_action = self.env.action_space.sample()

        self.env.seed(seed)
        self.reset()

    def reset(self):
        self.env.seed(random.randint(0, 99))
        self.time_step = 0
        self.total_reward = 0
        self.total_reward_shaped = 0
        self.random_action = self.env.action_space.sample()
        return self.preprocess_obs(self.env.reset())

    def preprocess_obs(self, obs):
        # if obs is image we want to convert
        # from hwc to cwh
        # since cwh is faster to process with cnn
        if self.obs_is_image:

            if self.obs_image_resize_to is not None:
                obs = cv2.resize(obs, tuple(self.obs_image_resize_to))

            if self.obs_image_to_grayscale:
                # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)[:, :, 2]
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            else:
                obs = np.transpose(obs, (2, 0, 1))

            return obs

        return obs

    def process_reward(self, reward):
        return reward * self.reward_scale

    def step(self, action):
        reward = 0.
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            r = self.process_reward(r)
            reward += r

            if done:
                break

        if self.visualize:
            if self.render_with_cv2:
                env_img = self.preprocess_obs(observation)
                if self.render_with_cv2_resize is not None:
                    env_img = cv2.resize(env_img, tuple(self.render_with_cv2_resize))
                cv2.imshow(
                    str(self.agent_id),
                    env_img
                )
                cv2.waitKey(3)
            else:
                self.env.render()

        reward_shaped = reward
        self.total_reward += reward
        self.total_reward_shaped += reward_shaped
        self.time_step += 1

        if self.time_step >= self.max_episode_length:
            done = True

        observation = self.preprocess_obs(observation)
        return observation, reward * self.reward_scale, done, info

    def get_step_count(self):
        return self.time_step

    def get_total_reward(self):
        return self.total_reward

    def get_total_reward_shaped(self):
        return self.total_reward_shaped

    def get_random_action(self, resample=False):
        if self.time_step % self.reinit_random_action_every == 0 or resample:
            self.random_action = self.env.action_space.sample()
        return self.random_action

    def get_logs(self):
        return {}
