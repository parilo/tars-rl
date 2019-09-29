import random
from threading import RLock

import numpy as np


class StartStatesBuffer:

    def __init__(
            self,
            capacity,
            observation_shapes,
            observation_dtypes,
            history_len
    ):
        self.size = capacity
        self.num_parts = len(observation_shapes)
        self.obs_shapes = observation_shapes
        self.obs_dtypes = observation_dtypes
        self.history_len = history_len

        self._store_lock = RLock()

        self.clear()

    def clear(self):
        with self._store_lock:
            self.num_in_buffer = 0
            self.stored_in_buffer = 0

            # initialize all np.arrays which store necessary data
            self.observations = []
            for part_id in range(self.num_parts):
                # print('--- ssb', part_id, (self.size,), self.obs_shapes[part_id], self.obs_dtypes[part_id],)
                obs = np.empty(
                    (self.size,) + (self.history_len[part_id],) + tuple(self.obs_shapes[part_id]),
                    dtype=self.obs_dtypes[part_id]
                )
                print('--- reserved for start states', obs.shape, obs.dtype)
                self.observations.append(obs)

            self.pointer = 0

    def push_episode(self, episode):
        """ episode = [observations, actions, rewards, dones]
            observations = [obs_part_1, ..., obs_part_n]
        """
        if len(episode[1]) == 0:
            print('--- warning: received zero length episode')
            return

        with self._store_lock:
            observations, actions, rewards, dones = episode

            for part_id in range(self.num_parts):
                self.observations[part_id][self.pointer] = np.array(observations[part_id][:self.history_len[part_id]])

            self.stored_in_buffer += 1
            self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
            self.pointer = (self.pointer + 1) % self.size

    def get_num_in_buffer(self):
        return self.num_in_buffer

    def get_batch(self, batch_size):

        with self._store_lock:

            indices = random.sample(range(self.num_in_buffer), k=batch_size)
            states = []
            for part_id in range(self.num_parts):
                state = [self.observations[part_id][indices[i]] for i in range(batch_size)]
                states.append(state)

            return states
