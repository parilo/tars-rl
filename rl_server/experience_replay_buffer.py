import random
import numpy as np
from collections import namedtuple


class ExperienceReplayBuffer ():

    def __init__(
        self,
        experience_replay_buffer_size,
        store_every_nth,
        num_of_parts_in_state
    ):
        self._experience_replay_buffer_size = experience_replay_buffer_size
        self._store_every_nth = store_every_nth
        self._num_of_parts_in_state = num_of_parts_in_state

        self._rewards = []
        self._actions = []
        self._terminators = []
        self._prev_states = [[] for i in range(self._num_of_parts_in_state)]
        self._next_states = [[] for i in range(self._num_of_parts_in_state)]

        self.store_index = 0
        self.store_invoked_count = store_every_nth
        self.stored_count = 0

        self.sum_rewards = 0

        self.transition = namedtuple(
            'Transition',
            ('s', 'a', 'r', 's_', 'done')
        )

    def get_buffer_size(self):
        return len(self._rewards)

    def get_stored_count(self):
        return self.stored_count

    def reset_sum_rewards(self):
        self.sum_rewards = 0

    def get_sum_rewards(self):
        return self.sum_rewards

    def store_exp(
        self,
        reward,
        action,
        prev_state,
        next_state,
        is_terminator
    ):
        self.store_exp_batch(
            [reward],
            [action],
            [prev_state],
            [next_state],
            [is_terminator]
        )

    def store_exp_batch(
        self,
        rewards,
        actions,
        prev_states,
        next_states,
        is_terminators
    ):

        if self.store_invoked_count % self._store_every_nth == 0:

            buffer_size = self.get_buffer_size()
            storing_count = len(rewards)

            if (
                (buffer_size + storing_count) <
                self._experience_replay_buffer_size
            ):
                self._rewards.extend(rewards)
                self._actions.extend(actions)
                self._terminators.extend(is_terminators)
                for i in range(self._num_of_parts_in_state):
                    self._prev_states[i].extend(prev_states[i])
                    self._next_states[i].extend(next_states[i])

            else:

                if buffer_size < self._experience_replay_buffer_size:
                    free_items_left = (
                        self._experience_replay_buffer_size -
                        self.stored_count
                    )
                    self._rewards.extend(rewards[:free_items_left])
                    self._actions.extend(actions[:free_items_left])
                    self._terminators.extend(is_terminators[:free_items_left])
                    for i in range(self._num_of_parts_in_state):
                        self._prev_states[i].extend(
                            prev_states[i][:free_items_left]
                        )
                        self._next_states[i].extend(
                            next_states[i][:free_items_left]
                        )

                    rewards = rewards[free_items_left:]
                    actions = actions[free_items_left:]
                    prev_states = prev_states[free_items_left:]
                    next_states = next_states[free_items_left:]
                    is_terminators = is_terminators[free_items_left:]

                    storing_count = len(rewards)

                self._rewards[
                    self.store_index:self.store_index + storing_count
                ] = rewards
                self._actions[
                    self.store_index:self.store_index + storing_count
                ] = actions
                self._terminators[
                    self.store_index:self.store_index + storing_count
                ] = is_terminators
                for i in range(self._num_of_parts_in_state):
                    self._prev_states[i][
                        self.store_index:self.store_index + storing_count
                    ] = prev_states[i]
                    self._next_states[i][
                        self.store_index:self.store_index + storing_count
                    ] = next_states[i]

            self.store_index = (
                self.store_index +
                storing_count) % self._experience_replay_buffer_size
            self.stored_count += storing_count
            self.sum_rewards += np.sum(rewards)

        self.store_invoked_count += 1

    def sample(self, batch_size):
        batch_indices = random.sample(
            range(self.get_buffer_size()),
            k=batch_size
        )
        a = np.array([self._actions[i] for i in batch_indices])
        r = np.array([self._rewards[i] for i in batch_indices])
        t = np.array([self._terminators[i] for i in batch_indices])
        s = []
        s_ = []
        for si in range(self._num_of_parts_in_state):
            s.append(np.array(
                [self._prev_states[si][i] for i in batch_indices]
            ))
            s_.append(np.array(
                [self._next_states[si][i] for i in batch_indices]
            ))

        batch = self.transition(s, a, r, s_, t)
        return batch
