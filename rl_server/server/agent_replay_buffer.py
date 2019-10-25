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
        self.discrete_actions = discrete_actions

        if discrete_actions:
            self.act_shape = (1,)
        else:
            self.act_shape = (action_size,)

        self.clear()

    def clear(self):
        self.observations = []
        for part_id in range(self.num_parts):
            self.observations.append(np.zeros(
                (self.size, ) + tuple(self.obs_shapes[part_id]),
                dtype=self.obs_dtypes[part_id]
            ))

        if self.discrete_actions:
            self.actions = np.zeros((self.size, ), dtype=np.int32)
        else:
            self.actions = np.zeros((self.size, ) + self.act_shape, dtype=np.float32)

        self.rewards = np.zeros((self.size, ), dtype=np.float32)
        self.dones = np.zeros((self.size, ), dtype=np.bool)
        self.inited = False

    def is_inited(self):
        return self.inited

    def push_init_observation(self, obs):
        for part_id in range(self.num_parts):
            self.observations[part_id][0] = obs[part_id]
        self.pointer = 0
        self.inited = True

    def get_last_obs(self):
        obs = []
        for part_id in range(self.num_parts):
            obs.append(self.observations[part_id][-1])
        return obs

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
        # indices = np.arange(self.pointer)
        observations = []
        for part_id in range(self.num_parts):
            observations.append(self.observations[part_id][:self.pointer])
        actions = self.actions[:self.pointer]
        rewards = self.rewards[:self.pointer]
        dones = self.dones[:self.pointer]
        return [observations, actions, rewards, dones]

    def get_episode_len(self):
        if hasattr(self, 'pointer'):
            return self.pointer
        else:
            return 0
