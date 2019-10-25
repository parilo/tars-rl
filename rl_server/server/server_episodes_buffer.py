import random
from threading import RLock

import numpy as np


class ServerEpisodesBuffer:

    def __init__(
        self,
        capacity,
    ):
        self.size = capacity
        self._store_lock = RLock()
        self.clear()

    def clear(self):
        with self._store_lock:
            self.stored_in_buffer = 0
            self.episodes = [None] * self.size
            # self.rewards = np.empty((self.size, ), dtype=np.float32)
            self.pointer = 0

    def push_episode(self, episode):
        """ episode = [observations, actions, rewards, dones]
            observations = [obs_part_1, ..., obs_part_n]
        """

        with self._store_lock:
            self.episodes[self.pointer] = episode
            # self.rewards[self.pointer] = np.sum(episode[2])
            self.pointer += 1
            self.pointer = self.pointer % self.size
            self.stored_in_buffer += 1

    def get_stored_in_buffer(self):
        return self.stored_in_buffer

    def get_batch(self, batch_size, indices=None):

        with self._store_lock:

            if indices is None:
                indices = random.sample(range(min(self.stored_in_buffer, self.size)), k=batch_size)

            batch_of_episodes = []
            for i in indices:
                batch_of_episodes.append(self.episodes[i])

            return batch_of_episodes
