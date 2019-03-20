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
        grayscale=False,
        **kwargs
    ):
        self._env = ObstacleTowerEnv(
            environment_filename,
            retro=retro,
            worker_id=agent_id
        )

        self._current_level = 0
        self._start_level_inc = start_level_inc
        self._grayscale = grayscale

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

    def process_obs(self, observation, info=None, reward=None):
        if info:
            vec_obs = np.copy(info['brain_info'].vector_observations[0])
            vec_obs[6] /= 3000.
            vec_obs = np.array(vec_obs.tolist() + [reward], dtype=np.float32)
        else:
            vec_obs = np.zeros((8,), dtype=np.float32)
            vec_obs[0] = 1
            vec_obs[6] = 1
        if self._grayscale:
            gs_img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            return [
                np.expand_dims(gs_img, axis=0),
                np.expand_dims(cv2.resize(gs_img, (32, 32)), axis=0),
                vec_obs
            ]
        else:
            return [
                np.transpose(cv2.resize(observation, (84, 84)), (2, 0, 1)),
                np.transpose(cv2.resize(observation, (32, 32)), (2, 0, 1)),
                vec_obs
            ]

    def render(self, obs):
        cv2.imshow(
            str(self.agent_id) + ' ' + str(obs[0].shape),
            cv2.resize(np.transpose(obs[0], (1, 2, 0)), (400, 400))
        )
        cv2.imshow(
            str(self.agent_id) + ' ' + str(obs[1].shape),
            cv2.resize(np.transpose(obs[1], (1, 2, 0)), (400, 400))
        )
        cv2.waitKey(3)

    def process_reward(self, reward):
        if reward == 1.:
            self._current_level += 1

        if reward > 0.05:
            return 1.
        else:
            return reward

    def step(self, action):
        observation, reward, done, info = super().step(action)

        obs = self.process_obs(observation, info, reward)
        if self.visualize:
            self.render(obs)

        # print(info['brain_info'].vector_observations)
        # if done:
        #     print('ep', self.time_step, self.total_reward, info['brain_info'].vector_observations)

        if obs[2][6] < 0.003:  # time left
            done = True

        return obs, reward, done, info

    def get_logs(self):
        logs = super().get_logs()
        logs.update({
            'level': self._current_level
        })
        return logs
