import numpy as np
import cv2
from obstacle_tower_env import ObstacleTowerEnv

from envs.gym_env import GymEnvWrapper


class ObstacleTowerEnvWrapper(GymEnvWrapper):

    def __init__(
        self,
        environment_filename,
        retro,
        agent_id,
        start_level_inc=None,
        **kwargs
    ):
        self._env = ObstacleTowerEnv(
            environment_filename,
            retro=retro,
            worker_id=agent_id
        )

        self._current_level = 0
        self._start_level_inc = start_level_inc

        super().__init__(env=self._env, agent_id=agent_id, **kwargs)

    def reset(self):
        if self._start_level_inc is None:
            self._current_level = 0
        else:
            self._current_level = min(max(0, self._current_level + self._start_level_inc), 24)
        self.env.floor(self._current_level)
        return self.process_obs(super().reset())

    # def split_observation(self, obs):
    #     obs_parts = []
    #     for i in range(3):
    #         for j in range(3):
    #             obs_parts.append(obs[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28])
    #     return np.stack(obs_parts, axis=0)
    #
    # def preprocess_obs(self, obs):
    #     obs = super().preprocess_obs(obs)
    #     return self.split_observation(obs)

    def process_obs(self, observation):
        return [
            observation,
            cv2.resize(observation, (28, 28))
        ]

    def render(self, obs):
        cv2.imshow(
            str(self.agent_id) + ' ' + str(obs[0].shape),
            cv2.resize(obs[0], (400, 400))
        )
        cv2.imshow(
            str(self.agent_id) + ' ' + str(obs[1].shape),
            cv2.resize(obs[1], (400, 400))
        )
        cv2.waitKey(3)

    def step(self, action):
        observation, reward, done, info = super().step(action)

        if reward == 1.:
            self._current_level += 1

        obs = self.process_obs(observation)
        if self.visualize:
            self.render(obs)

        return obs, reward, done, info

    def get_logs(self):
        logs = super().get_logs()
        logs.update({
            'level': self._current_level
        })
        return logs
