import numpy as np
import cv2
import gym

from envs.gym_env import GymEnvWrapper


class CarRacingWrapper(GymEnvWrapper):

    def __init__(
        self,
        agent_id,
        **kwargs
    ):
        self._env = gym.make('CarRacing-v0')
        super().__init__(env=self._env, agent_id=agent_id, **kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # cv2.imshow(
        #     str(self.agent_id) + ' ' + str(observation.shape),
        #     cv2.resize(observation, (400, 400))
        # )
        # cv2.waitKey(3)
        # print(reward)

        return observation, reward, done, info
