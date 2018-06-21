import random
import numpy as np
from collections import namedtuple
from threading import Lock


class ServerBuffer:

    def __init__(self, capacity, observation_shapes, action_size):
        self.size = capacity
        self.num_in_buffer = 0
        self.num_parts = len(observation_shapes)
        self.obs_shapes = observation_shapes
        self.act_shape = (action_size,)

        # initialize all np.arrays which store necessary data
        self.observations = []
        for part_id in range(self.num_parts):
            self.observations.append(np.empty((self.size, ) + self.obs_shapes[part_id], dtype=np.float32))
        self.actions = np.empty((self.size, ) + self.act_shape, dtype=np.float32)
        self.rewards = np.empty((self.size, ), dtype=np.float32)
        self.dones = np.empty((self.size, ), dtype=np.bool)
        self.td_errors = np.empty((self.size, ), dtype=np.float32)

        self.pointer = 0
        self._store_lock = Lock()
        self.transition = namedtuple('Transition', ('s', 'a', 'r', 's_', 'done'))

    def push_episode(self, episode):
        """ episode = [observations, actions, rewards, dones]
            observations = [obs_part_1, ..., obs_part_n]
        """

        with self._store_lock:

            observations, actions, rewards, dones = episode
            episode_len = len(actions)
            self.num_in_buffer += episode_len
            self.num_in_buffer = min(self.size, self.num_in_buffer)

            indices = np.arange(self.pointer, self.pointer + episode_len) % self.size
            for part_id in range(self.num_parts):
                self.observations[part_id][indices] = np.array(observations[part_id])
            self.actions[indices] = np.array(actions)
            self.rewards[indices] = np.array(rewards)
            self.dones[indices] = np.array(dones)
            self.td_errors[indices] = np.ones(len(indices))

            self.pointer = (self.pointer + episode_len) % self.size

    def get_transition(self, idx, history_len=1):

        state, next_state = [], []
        for part_id in range(self.num_parts):

            s = np.zeros((history_len, ) + self.obs_shapes[part_id], dtype=np.float32)
            indices = [idx]
            for i in range(history_len-1):
                if (self.num_in_buffer == self.size):
                    next_idx = (idx-i-1) % self.size
                else:
                    next_idx = idx-i-1
                if (next_idx < 0 or self.dones[next_idx]):
                    break
                indices.append(next_idx)
            indices = indices[::-1]
            s[-len(indices):] = self.observations[part_id][indices]
            state.append(s)

            s_ = np.zeros_like(s)
            indices = indices[1:]
            if not self.dones[idx]:
                indices.append((idx + 1) % self.size)
            s_[:len(indices)] = self.observations[part_id][indices]
            next_state.append(s_)

        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        td_error = self.td_errors[idx]
        return state, action, reward, next_state, done, td_error

    def update_td_errors(self, indices, td_errors):
        self.td_errors[indices] = td_errors

    def get_batch(self, batch_size, history_len=1, indices=None):

        if indices is None:
            indices = random.sample(range(self.num_in_buffer), k=batch_size)

        transitions = []
        for idx in indices:
            transitions.append(self.get_transition(idx, history_len))

        states = []
        for part_id in range(self.num_parts):
            state = np.array([transitions[i][0][part_id] for i in range(batch_size)])
            states.append(state)
        actions = np.array([transitions[i][1] for i in range(batch_size)])
        rewards = np.array([transitions[i][2] for i in range(batch_size)])
        next_states = []
        for part_id in range(self.num_parts):
            next_state = np.array([transitions[i][3][part_id] for i in range(batch_size)])
            next_states.append(next_state)
        dones = np.array([transitions[i][4] for i in range(batch_size)])
        batch = self.transition(states, actions, rewards, next_states, dones)
        return batch
    
    def get_prioritized_batch(self, batch_size, history_len=1, 
                              priority='proportional', alpha=1.0, beta=1.0):

        if priority == 'proportional':
            p = np.power(np.abs(self.td_errors[:self.num_in_buffer]), alpha)
            p = p / p.sum()
            indices = np.random.choice(range(self.num_in_buffer), size=batch_size, p=p)
            probs = p[indices]
            is_weights = np.power(self.num_in_buffer * probs, -beta)
            is_weights = is_weights / is_weights.max()

        batch = self.get_batch(batch_size, history_len, indices)
        return batch, indices, is_weights
