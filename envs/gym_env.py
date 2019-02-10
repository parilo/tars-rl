import numpy as np

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
        obs_is_image=False
    ):
        self.env = env
        self.reward_scale = reward_scale
        self.frame_skip = frame_skip
        self.visualize = visualize
        self.seed = seed
        self.max_episode_length = max_episode_length
        self.obs_is_image = obs_is_image

        self.reinit_random_action_every = reinit_random_action_every
        self.random_action = self.env.action_space.sample()

        self.env.seed(seed)
        self.reset()

    def reset(self):
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
            im = np.transpose(obs, (2, 0, 1))
            return im

        return obs

    def step(self, action):
        reward = 0.
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            reward += r

            if self.visualize:
                self.env.render()

            if done: break

        reward_shaped = reward * self.reward_scale
        self.total_reward += reward
        self.total_reward_shaped += reward_shaped
        self.time_step += 1

        if self.time_step >= self.max_episode_length:
            done = True

        observation = self.preprocess_obs(observation)
        return observation, reward * self.reward_scale, done, info

    def get_total_reward(self):
        return self.total_reward

    def get_total_reward_shaped(self):
        return self.total_reward_shaped

    def get_random_action(self, resample=False):
        if self.time_step % self.reinit_random_action_every == 0 or resample:
            self.random_action = self.env.action_space.sample()
        return self.random_action
