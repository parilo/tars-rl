import numpy as np


class AgentBuffer:

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

        if discrete_actions:
            self.act_shape = (1,)
        else:
            self.act_shape = (action_size,)

        self.observations = []
        for part_id in range(self.num_parts):
            self.observations.append(np.empty(
                (self.size, ) + tuple(self.obs_shapes[part_id]),
                dtype=self.obs_dtypes[part_id]
            ))

        if discrete_actions:
            self.actions = np.empty((self.size, ), dtype=np.int32)
        else:
            self.actions = np.empty((self.size, ) + self.act_shape, dtype=np.float32)

        self.rewards = np.empty((self.size, ), dtype=np.float32)
        self.dones = np.empty((self.size, ), dtype=np.bool)
        self.inited = False

    def is_inited(self):
        return self.inited

    def push_init_observation(self, obs):
        for part_id in range(self.num_parts):
            self.observations[part_id][0] = obs[part_id]
        self.pointer = 0
        self.inited = True

    def get_current_state(self, history_len=1):
        state = []
        for part_id, part_hist_len in zip(range(self.num_parts), history_len):
            s = np.zeros((part_hist_len, ) + tuple(self.obs_shapes[part_id]), dtype=np.float32)
            indices = np.arange(max(0, self.pointer - part_hist_len + 1), self.pointer + 1)
            s[-len(indices):] = self.observations[part_id][indices]
            state.append(s)
        return state

    def push_transition(self, transition):
        """ transition = [next_obs, action, reward, done]
            next_obs = [next_obs_part_1, ..., next_obs_part_n]
        """
        next_obs, action, reward, done = transition
        for part_id in range(self.num_parts):
            self.observations[part_id][self.pointer + 1] = next_obs[part_id]
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer += 1

    def get_complete_episode(self):
        indices = np.arange(self.pointer)
        observations = []
        for part_id in range(self.num_parts):
            observations.append(self.observations[part_id][indices])
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        return [observations, actions, rewards, dones]
