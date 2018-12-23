class GymEnvWrapper:

    def __init__(
        self,
        env,
        reward_scale=1.,
        frame_skip=1,
        visualize=False,
        reinit_random_action_every=1
    ):
        self.env = env
        self.reward_scale = reward_scale
        self.frame_skip = frame_skip
        self.visualize = visualize

        self.reinit_random_action_every = reinit_random_action_every
        self.random_action = self.env.action_space.sample()

        self.reset()

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.total_reward_shaped = 0
        self.random_action = self.env.action_space.sample()
        return self.env.reset()

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
        return observation, reward * self.reward_scale, done, info

    def get_total_reward(self):
        return self.total_reward

    def get_total_reward_shaped(self):
        return self.total_reward_shaped

    def get_random_action(self, resample=True):
        if self.time_step % self.reinit_random_action_every == 0 or resample:
            self.random_action = self.env.action_space.sample()
        return self.random_action
