import random
import time
import numpy as np
from collections import namedtuple
from threading import RLock


class ServerBuffer:

    def __init__(self, capacity, observation_shapes, action_size):
        self.size = capacity
        self.num_in_buffer = 0
        self.stored_in_buffer = 0
        self.num_parts = len(observation_shapes)
        self.obs_shapes = observation_shapes
        self.act_shape = (action_size,)

        # initialize all np.arrays which store necessary data
        self.observations = []
        for part_id in range(self.num_parts):
            obs = np.empty((self.size, ) + self.obs_shapes[part_id], dtype=np.float32)
            self.observations.append(obs)
        self.actions = np.empty((self.size, ) + self.act_shape, dtype=np.float32)
        self.rewards = np.empty((self.size, ), dtype=np.float32)
        self.dones = np.empty((self.size, ), dtype=np.bool)
        self.td_errors = np.empty((self.size, ), dtype=np.float32)

        self.pointer = 0
        self._store_lock = RLock()
        self.transition = namedtuple('Transition', ('s', 'a', 'r', 's_', 'done'))

    def push_episode(self, episode):
        """ episode = [observations, actions, rewards, dones]
            observations = [obs_part_1, ..., obs_part_n]
        """

        with self._store_lock:

            observations, actions, rewards, dones = episode
            episode_len = len(actions)
            self.stored_in_buffer += episode_len
            self.num_in_buffer = min(self.size, self.num_in_buffer + episode_len)

            indices = np.arange(self.pointer, self.pointer + episode_len) % self.size
            for part_id in range(self.num_parts):
                self.observations[part_id][indices] = np.array(observations[part_id])
            self.actions[indices] = np.array(actions)
            self.rewards[indices] = np.array(rewards)
            self.dones[indices] = np.array(dones)
            self.td_errors[indices] = np.ones(len(indices))

            self.pointer = (self.pointer + episode_len) % self.size

    def get_stored_in_buffer(self):
        return self.stored_in_buffer

    def get_state(self, idx, history_len=1):
        """ compose the state from a number (history_len) of observations
        """
        state = []
        for part_id in range(self.num_parts):
            start_idx = idx - history_len + 1

            if (start_idx < 0 or np.any(self.dones[start_idx:idx+1])):
                s = np.zeros((history_len, ) + self.obs_shapes[part_id], dtype=np.float32)
                indices = [idx]
                for i in range(history_len-1):
                    next_idx = (idx-i-1) % self.size
                    if next_idx >= self.num_in_buffer or self.dones[next_idx]:
                        break
                    indices.append(next_idx)
                indices = indices[::-1]
                s[-len(indices):] = self.observations[part_id][indices]
            else:
                s = self.observations[part_id][slice(start_idx, idx+1, 1)]
                
            state.append(s)
        return state

    def get_transition_n_step(self, idx, history_len=1, n_step=1, gamma=0.99):
        state = self.get_state(idx, history_len)
        next_state = self.get_state((idx + n_step) % self.size, history_len)
        cum_reward = 0
        indices = np.arange(idx, idx + n_step) % self.size
        for num, i in enumerate(indices):
            cum_reward += self.rewards[i] * (gamma ** num)
            done = self.dones[i]
            if done:
                break
        return state, self.actions[idx], cum_reward, next_state, done, self.td_errors[idx]

    def update_td_errors(self, indices, td_errors):
        self.td_errors[indices] = td_errors

    def get_batch(self, batch_size, history_len=1, n_step=1, gamma=0.99, indices=None):

        with self._store_lock:

            if indices is None:
                indices = random.sample(range(self.num_in_buffer), k=batch_size)
            transitions = []
            for idx in indices:
                transition = self.get_transition_n_step(idx, history_len, n_step, gamma)
                transitions.append(transition)

            states = []
            for part_id in range(self.num_parts):
                state = [transitions[i][0][part_id] for i in range(batch_size)]
                states.append(state)
            actions = [transitions[i][1] for i in range(batch_size)]
            rewards = [transitions[i][2] for i in range(batch_size)]
            next_states = []
            for part_id in range(self.num_parts):
                next_state = [transitions[i][3][part_id] for i in range(batch_size)]
                next_states.append(next_state)
            dones = [transitions[i][4] for i in range(batch_size)]

            batch = self.transition(states, actions, rewards, next_states, dones)
            return batch

    def get_prioritized_batch(self, batch_size, history_len=1,
                              n_step=1, gamma=0.99,
                              priority='proportional', alpha=0.6, beta=1.0):

        with self._store_lock:

            if priority == 'proportional':
                p = np.power(np.abs(self.td_errors[:self.num_in_buffer])+1e-6, alpha)
                p = p / p.sum()
                indices = np.random.choice(range(self.num_in_buffer), size=batch_size, p=p)
                probs = p[indices]
                is_weights = np.power(self.num_in_buffer * probs, -beta)
                is_weights = is_weights / is_weights.max()

            batch = self.get_batch(batch_size, history_len, n_step, gamma, indices)
            return batch, indices, is_weights
