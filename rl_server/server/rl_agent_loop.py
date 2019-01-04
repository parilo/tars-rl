import random
import copy

import numpy as np

from rl_server.server.rl_client import RLClient
from rl_server.server.agent_replay_buffer import AgentBuffer
# from misc.defaults import set_agent_seed, init_episode_storage
from misc.common import set_global_seeds
from misc.rl_logger import RLLogger

np.set_printoptions(suppress=True)


class RLAgent:
    
    def __init__(
        self,
        env,
        exp_config,
        agent_config
    ):
        self._env = env
        self._exp_config = exp_config
        self._logdir = exp_config.server.logdir
        self._validation = agent_config.exploration is None
        self._exploration = copy.deepcopy(agent_config.exploration)
        self._step_limit = exp_config.env.step_limit
        self._store_episodes = agent_config.store_episodes
        self._id = agent_config.agent_id
        self._seed = agent_config.seed
        self._history_len = exp_config.env.history_length
        self._algorithm_id = agent_config.algorithm_id

        # exploration
        if self._exploration is not None:
            if hasattr(self._exploration, 'normal_noise'):
                # exploration with normal noise
                self._exploration.normal_noise = np.float(self._exploration.normal_noise)
            else:
                self._exploration.normal_noise = None

            if hasattr(self._exploration, 'random_action_prob'):
                # exploration with random action
                self._exploration.random_action_prob = np.float(self._exploration.random_action_prob)
            else:
                self._exploration.random_action_prob = None

        # action remap
        self._action_remap_function = None
        if hasattr(exp_config.env, 'remap_action'):
            low_from = exp_config.env.remap_action.low.before
            low_to = exp_config.env.remap_action.low.after
            high_from = exp_config.env.remap_action.high.before
            high_to = exp_config.env.remap_action.high.after
            def remap_func (action):
                return (action - low_from) / (high_from - low_from) * (high_to - low_to) + low_to
            self._action_remap_function = remap_func

        # action clipping function
        self._clipping_function = None
        if exp_config.actor.output_activation == 'tanh':
            def tanh_clipping(action):
                return np.clip(action, -1., 1.)
            self._clipping_function = tanh_clipping

        (
            self._observation_shapes,
            self._state_shapes,
            self._action_size
        ) = self._exp_config.get_env_shapes()

        set_global_seeds(self._seed)
        self._logger = RLLogger(
            self._logdir,
            self._id,
            self._validation,
            self._env
        )

        self._rl_client = RLClient(
            port=self._exp_config.server.client_start_port + self._id
        )

        if self._exp_config.framework == 'tensorflow':
            from rl_server.tensorflow.agent_model_tf import AgentModel
            self._agent_model = AgentModel(
                self._exp_config,
                self._rl_client
            )
            self.fetch_model()

    def fetch_model(self):
        # self._agent_model.fetch(self._id % self._algos_count)
        self._agent_model.fetch(self._algorithm_id)

    def init_agent_buffers(self):
        buf_capacity = self._exp_config.env.agent_buffer_size
        first_obs = self._env.reset()

        self._agent_buffer = AgentBuffer(
            buf_capacity,
            self._observation_shapes,
            self._action_size
        )
        self._agent_buffer.push_init_observation([first_obs])

        # self.augmented_agent_buffers = []
        # for _ in range(self._num_of_augmented_targets):
        #     aug_agent_buffer = AgentBuffer(
        #         buf_capacity,
        #         self._env.observation_shapes,
        #         self._env.action_size)
        #     aug_agent_buffer.push_init_observation([first_obs])
        #     self.augmented_agent_buffers.append(aug_agent_buffer)

    def run(self):

        self.init_agent_buffers()

        # if self._store_episodes:
        #     path_to_episode_storage, stored_episode_index = init_episode_storage(self._id, self._logdir)
        #     print('stored episodes: {}'.format(stored_episode_index))

        n_steps = 0
        episode_index = 0

        # exploration parameters for gradient exploration
        # explore_start_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_end_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_episodes = 500
        # explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
        # explore_temp = explore_start_temp
        
        def prepare_state(state):
            return [np.expand_dims(state, axis=0)]

        while True:

            # obtain current state from the buffer
            state = self._agent_buffer.get_current_state(
                history_len=self._history_len
            )[0]

            # Bernoulli exploration
            # action = np.array(self._agent_model.act_batch(prepare_state(state))[0])
            # env_action = (action + 1.) / 2.
            # env_action = np.clip(env_action, 0., 1.)
            # env_action = np.random.binomial([1]*self._env.action_size, env_action).astype(np.float32)

            if self._exploration.built_in_algo:
                if self._exploration.validation:
                    action = self._agent_model.act_batch(
                        prepare_state(state),
                        mode='deterministic'
                    )[0]
                else:
                    action = self._agent_model.act_batch(prepare_state(state))[0]
                action = np.array(action)

            else:
                action = self._agent_model.act_batch(prepare_state(state))[0]
                action = np.array(action)

                if not self._validation:
                    if self._exploration.normal_noise is not None:
                        # exploration with normal noise
                        action += np.random.normal(
                           scale=self._exploration.normal_noise,
                           size=self._action_size
                        )
                    if self._exploration.random_action_prob is not None:
                        if random.random() < self._exploration.random_action_prob:
                            action = self._env.get_random_action()

                    action = self._clipping_function(action)

            # action remap function
            if self._action_remap_function is not None:
                env_action = self._action_remap_function(action)
            else:
                env_action = action

            next_obs, reward, done, info = self._env.step(env_action)
            transition = [[next_obs], action, reward, done]
            self._agent_buffer.push_transition(transition)
            
            # for i in range(self._num_of_augmented_targets):
            #     augmented_reward = info['augmented_targets']['rewards'][i]
            #     augmented_obs = info['augmented_targets']['observations'][-1][i]
            #     aug_transition = [[augmented_obs], action, augmented_reward, done]
            #     self.augmented_agent_buffers[i].push_transition(aug_transition)
            
            # next_state = self.agent_buffer.get_current_state(
            #     history_len=self._history_len
            # )[0].ravel()
            n_steps += 1

            if done or (self._step_limit > 0 and n_steps > self._step_limit):

                self._logger.log(episode_index, n_steps)
                episode = self._agent_buffer.get_complete_episode()
                self._rl_client.store_episode(episode)

                # for augmented_agent_buffer in self.augmented_agent_buffers:
                #     aug_episode = augmented_agent_buffer.get_complete_episode()
                #     self._rl_client.store_episode(aug_episode)
                
                self.fetch_model()

                # save episode on disk
                # if self._store_episodes:
                #     ep_path = os.path.join(path_to_episode_storage, 'episode_' + str(stored_episode_index)+'.pkl')
                #     with open(ep_path, 'wb') as f:
                #         pickle.dump(
                #         [
                #             [obs_part.tolist() for obs_part in episode[0]],
                #             episode[1].tolist(),
                #             episode[2].tolist(),
                #             episode[3].tolist()
                #         ], f, pickle.HIGHEST_PROTOCOL)
                #     stored_episode_index += 1

                episode_index += 1
                self.init_agent_buffers()
                n_steps = 0