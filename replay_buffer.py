import numpy as np
from collections import namedtuple
from threading import Lock

#############################################################################################################
############################################# Huge server buffer ############################################
#############################################################################################################


class ServerBuffer:

    def __init__(self, capacity, history_len=1, num_of_parts_in_obs=1):
        self.size = capacity
        self.hist_len = history_len
        self.num_in_buffer = 0
        self.num_parts = num_of_parts_in_obs
        self.observations = [None for part_id in range(self.num_parts)]
        self.actions = None
        self.rewards = None
        self.dones = None
        self._store_lock = Lock()
        self.transition = namedtuple('Transition', ('s', 'a', 'r', 's_', 'done'))

    def push_episode(self, episode):
        """episode = [observations, actions, rewards, dones]
        """

        with self._store_lock:
            observations, actions, rewards, dones = episode
            self.obs_shapes = [observations[part_id][0].shape for part_id in range(self.num_parts)]
            self.act_shape = actions[0].shape
            episode_len = len(actions)
            self.num_in_buffer += episode_len
            self.num_in_buffer = min(self.size, self.num_in_buffer)

            if self.actions is None:
                for part_id in range(self.num_parts):
                    self.observations[part_id] = np.empty(
                        (self.size, ) + self.obs_shapes[part_id], dtype=np.float32)
                self.actions = np.empty((self.size, ) + self.act_shape, dtype=np.float32)
                self.rewards = np.empty((self.size, ), dtype=np.float32)
                self.dones = np.empty((self.size, ), dtype=np.bool)
                self.pointer = 0

            indices = np.arange(self.pointer, self.pointer + episode_len) % self.size
            for part_id in range(self.num_parts):
                self.observations[part_id][indices] = np.array(observations[part_id])
            self.actions[indices] = np.array(actions)
            self.rewards[indices] = np.array(rewards)
            self.dones[indices] = np.array(dones)

            self.pointer = (self.pointer + episode_len) % self.size

    def get_transition(self, idx):

        state, next_state = [], []
        for part_id in range(self.num_parts):
            s = np.zeros((self.hist_len, ) + self.obs_shapes[part_id], dtype=np.float32)
            s[-1] = self.observations[part_id][idx]
            for i in range(self.hist_len-1):
                if (idx-i-1 < 0 or self.dones[idx-i-1]): break
                s[-2-i] = self.observations[part_id][idx-i-1]

            s_ = np.zeros_like(s)
            s_[:-1] = s[1:]
            if not self.dones[idx]:
                s_[-1] = self.observations[part_id][(idx + 1) % self.size]

            state.append(s)
            next_state.append(s_)

        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]

        return state, action, reward, next_state, done

    def get_random_indices(self, num_indices):
        indices = np.arange(self.num_in_buffer)
        np.random.shuffle(indices)
        return indices[:num_indices]

    def get_batch(self, batch_size):
        indices = self.get_random_indices(batch_size)
        transitions = []
        for idx in indices:
            transitions.append(self.get_transition(idx))

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

#############################################################################################################
###################################### Small local buffer of each agent #####################################
#############################################################################################################


class AgentBuffer:

    def __init__(self, capacity, history_len=1, num_of_parts_in_obs=1):
        self.size = capacity
        self.hist_len = history_len
        self.num_parts = num_of_parts_in_obs
        self.observations = [None for part_id in range(self.num_parts)]
        self.actions = None
        self.rewards = None
        self.dones = None

    def push_init_observation(self, obs):
        for part_id in range(self.num_parts):
            self.observations[part_id] = np.empty((self.size, ) + obs[part_id].shape, dtype=np.float32)
            self.observations[part_id][:self.hist_len-1] = np.zeros((self.hist_len-1,) + obs[part_id].shape)
            self.observations[part_id][self.hist_len-1] = obs[part_id]
        self.pointer = self.hist_len-1

    def get_current_state(self):
        state = []
        for part_id in range(self.num_parts):
            indices = range(self.pointer-self.hist_len+1, self.pointer)
            s = self.observations[part_id][indices]
            state.append(s)
        return state

    def push_transition(self, transition):
        """ transition = [next_obs, action, reward, done]
        """
        next_obs, action, reward, done = transition
        if self.actions is None:
            self.actions = np.empty((self.size, ) + action.shape, dtype=np.float32)
            self.rewards = np.empty((self.size, ), dtype=np.float32)
            self.dones = np.empty((self.size, ), dtype=np.bool)

        for part_id in range(self.num_parts):
            self.observations[part_id][self.pointer+1] = next_obs[part_id]
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer += 1

    def get_complete_episode(self):
        indices = np.arange(self.hist_len-1, self.pointer)
        observations = []
        for part_id in range(self.num_parts):
            observations.append(self.observations[part_id][indices])
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        return [observations, actions, rewards, dones]
