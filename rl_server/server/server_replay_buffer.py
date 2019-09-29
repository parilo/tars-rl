import random
from collections import namedtuple
from threading import RLock

import numpy as np


Transition = namedtuple("Transition", ("s", "a", "r", "s_", "done", "valid_mask", "next_valid_mask"))


class ServerBuffer:

    def __init__(
        self,
        capacity,
        observation_shapes,
        observation_dtypes,
        action_size,
        discrete_actions=False
    ):
        self.size = capacity
        self.num_parts = len(observation_shapes)
        self.obs_shapes = observation_shapes
        self.obs_dtypes = observation_dtypes
        self.discrete_actions = discrete_actions
        self.action_size = action_size

        self._store_lock = RLock()

        self.clear()

    def clear(self):
        with self._store_lock:
            self.num_in_buffer = 0
            self.stored_in_buffer = 0

            if self.discrete_actions:
                self.act_shape = (1,)
            else:
                self.act_shape = (self.action_size,)

            # initialize all np.arrays which store necessary data
            self.observations = []
            for part_id in range(self.num_parts):
                print(part_id, (self.size, ), self.obs_shapes[part_id], self.obs_dtypes[part_id])
                obs = np.empty(
                    (self.size, ) + tuple(self.obs_shapes[part_id]),
                    dtype=self.obs_dtypes[part_id]
                )
                # print('created obs array', obs.nbytes / 1024 / 1024, obs.shape, obs.dtype)
                self.observations.append(obs)

            if self.discrete_actions:
                self.actions = np.empty((self.size, ), dtype=np.int32)
            else:
                self.actions = np.empty((self.size, ) + self.act_shape, dtype=np.float32)

            self.rewards = np.empty((self.size, ), dtype=np.float32)
            self.dones = np.empty((self.size, ), dtype=np.bool)
            self.td_errors = np.empty((self.size, ), dtype=np.float32)

            self.pointer = 0

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

    def _get_indices_backward(self, start_index, history_len):
        indices = [start_index]
        for i in range(history_len - 1):
            next_idx = (start_index - i - 1) % self.size
            if next_idx >= self.num_in_buffer or self.dones[next_idx]:
                break
            indices.append(next_idx)
        return indices[::-1]

    def get_state(self, idx, history_len=1):
        """ compose the state from a number (history_len) of observations
        """
        state = []
        valid_masks = []  # transitions before reached done are not valid
        for part_id, part_hist_len in zip(range(self.num_parts), history_len):
            start_idx = idx - part_hist_len + 1

            if (start_idx < 0 or np.any(self.dones[start_idx:idx + 1])):
                s = np.zeros((part_hist_len, ) + tuple(self.obs_shapes[part_id]), dtype=np.float32)
                v_mask = np.zeros((part_hist_len, ), dtype=np.float32)
                indices = self._get_indices_backward(idx, part_hist_len)
                # indices = [idx]
                # for i in range(part_hist_len - 1):
                #     next_idx = (idx-i-1) % self.size
                #     if next_idx >= self.num_in_buffer or self.dones[next_idx]:
                #         break
                #     indices.append(next_idx)
                # indices = indices[::-1]
                s[-len(indices):] = self.observations[part_id][indices]
                v_mask[-len(indices):] = 1
            else:
                s = self.observations[part_id][slice(start_idx, idx+1, 1)]
                v_mask = np.ones((part_hist_len, ), dtype=np.float32)
            state.append(s)
            valid_masks.append(v_mask)
        return state, valid_masks

    def get_action_history(self, idx, history_len=1):
        start_idx = idx - history_len + 1

        if (start_idx < 0 or np.any(self.dones[start_idx:idx + 1])):
            actions = np.zeros((history_len,) + (self.action_size,), dtype=np.float32)
            indices = self._get_indices_backward(idx, history_len)
            # indices = [idx]
            # for i in range(history_len - 1):
            #     next_idx = (idx - i - 1) % self.size
            #     if next_idx >= self.num_in_buffer or self.dones[next_idx]:
            #         break
            #     indices.append(next_idx)
            # indices = indices[::-1]
            actions[-len(indices):] = self.actions[indices]
        else:
            actions = self.actions[slice(start_idx, idx + 1, 1)]

        return actions

    def get_transition_n_step(self, idx, history_len=1, n_step=1, gamma=0.99, action_history=False):
        state, valid_masks = self.get_state(idx, history_len)
        next_state, next_valid_masks = self.get_state((idx + n_step) % self.size, history_len)
        cum_reward = 0
        indices = np.arange(idx, idx + n_step) % self.size
        for num, i in enumerate(indices):
            cum_reward += self.rewards[i] * (gamma ** num)
            done = self.dones[i]
            if done:
                break
        if action_history:
            actions = self.get_action_history(idx, history_len[0])  # action history uses history len from the first obs
        else:
            actions = self.actions[idx]
        return state, actions, cum_reward, next_state, done, self.td_errors[idx], valid_masks, next_valid_masks

    def update_td_errors(self, indices, td_errors):
        self.td_errors[indices] = td_errors

    def get_batch(self, batch_size, history_len=1, n_step=1, gamma=0.99, indices=None, action_history=False):

        with self._store_lock:

            if indices is None:
                indices = random.sample(range(self.num_in_buffer - max(history_len)), k=batch_size)
            transitions = []
            for idx in indices:
                transition = self.get_transition_n_step(idx, history_len, n_step, gamma, action_history)
                transitions.append(transition)

            states = []
            next_states = []
            valid_masks = []
            next_valid_masks = []
            for part_id in range(self.num_parts):
                state = [transitions[i][0][part_id] for i in range(batch_size)]
                states.append(state)
                next_state = [transitions[i][3][part_id] for i in range(batch_size)]
                next_states.append(next_state)
                valid_masks.append([transitions[i][6][part_id] for i in range(batch_size)])
                next_valid_masks.append([transitions[i][7][part_id] for i in range(batch_size)])

            actions = [transitions[i][1] for i in range(batch_size)]
            rewards = [transitions[i][2] for i in range(batch_size)]

            dones = [transitions[i][4] for i in range(batch_size)]

            batch = Transition(
                [np.array(state_part, dtype=np.float32) for state_part in states],
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                [np.array(next_state_part, dtype=np.float32) for next_state_part in next_states],
                np.array(dones, dtype=np.bool),
                [np.array(mask, dtype=np.float32) for mask in valid_masks],
                [np.array(mask, dtype=np.float32) for mask in next_valid_masks],
            )
            return batch

    def get_prioritized_batch(self, batch_size, history_len=1,
                              n_step=1, gamma=0.99,
                              priority="proportional", alpha=0.6, beta=1.0):

        with self._store_lock:

            if priority == "proportional":
                p = np.power(np.abs(self.td_errors[:self.num_in_buffer])+1e-6, alpha)
                p = p / p.sum()
                indices = np.random.choice(range(self.num_in_buffer), size=batch_size, p=p)
                probs = p[indices]
                is_weights = np.power(self.num_in_buffer * probs, -beta)
                is_weights = is_weights / is_weights.max()

            batch = self.get_batch(batch_size, history_len, n_step, gamma, indices)
            return batch, indices, is_weights
