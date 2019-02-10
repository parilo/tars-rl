import random
import copy
import os
import pickle

import numpy as np

from rl_server.server.rl_client import RLClient
from rl_server.server.agent_replay_buffer import AgentBuffer
from misc.common import set_global_seeds, create_if_need
from misc.rl_logger import RLLogger

np.set_printoptions(suppress=True)


class RLAgent:
    
    def __init__(
        self,
        env,
        exp_config,
        agent_config,
        checkpoint_path=None
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
        self._checkpoint_path = checkpoint_path
        self._repeat_action = agent_config.repeat_action
        self._random_repeat_action = agent_config.random_repeat_action

        # exploration
        if self._exploration is not None:
            if self._exploration.isset('normal_noise'):
                # exploration with normal noise
                self._exploration.normal_noise = np.float(self._exploration.normal_noise)
            else:
                self._exploration.normal_noise = None

            if self._exploration.isset('random_action_prob'):
                # exploration with random action
                self._exploration.random_action_prob = np.float(self._exploration.random_action_prob)
            else:
                self._exploration.random_action_prob = None

        if hasattr(self._exploration, 'built_in_algo') and self._exploration.built_in_algo:
            self._validation = self._exploration.validation

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
        def trivial_clipping(action):
            return action
        self._clipping_function = trivial_clipping
        if (
            (
                exp_config.actor.isset('output_activation') and
                exp_config.actor.output_activation == 'tanh'
            ) or (
                exp_config.env.isset('clip_action') and
                exp_config.env.clip_action == 'tanh'
            )
        ):
            def tanh_clipping(action):
                return np.clip(action, -1., 1.)
            self._clipping_function = tanh_clipping

        # action postprocess
        def trivial_action_proctprocess(action):
            return action
        self._action_postprocess = trivial_action_proctprocess
        if exp_config.env.isset('action_postprocess'):

            if exp_config.env.action_postprocess.type == 'argmax of softmax':
                from scipy.special import softmax

                boltzmann_expl = (
                    self._exploration is not None and
                    self._exploration.isset('type') and
                    self._exploration.type == 'boltzmann'
                )
                if boltzmann_expl:
                    possible_actions = list(range(0, self._exp_config.env.action_size))

                def softmax_action_postprocess(action):
                    if boltzmann_expl:
                        return np.random.choice(possible_actions, p=softmax(action))
                    else:
                        return np.argmax(softmax(action))
                    # if boltzmann_expl:
                    #     return np.random.choice(possible_actions, p=np.array(action)/np.sum(action))
                    # else:
                    #     return np.argmax(action)

                self._action_postprocess = softmax_action_postprocess

        # store episodes
        self._store_episodes = False
        if hasattr(agent_config, 'store_episodes') and agent_config.store_episodes:
            self._store_episodes = True

        (
            self._observation_shapes,
            self._observation_dtypes,
            self._state_shapes,
            self._action_size
        ) = self._exp_config.get_env_shapes()

        set_global_seeds(self._seed)
        self._logger = RLLogger(
            self._logdir,
            self._id,
            self._validation,
            self._env,
            exp_config.env.log_every_n_steps
        )

        self._rl_client = None
        if self._checkpoint_path is None:
            self._rl_client = RLClient(
                port=self._exp_config.server.client_start_port + self._id
            )

        if self._exp_config.framework == 'tensorflow':
            from rl_server.tensorflow.agent_model_tf import AgentModel
            self._agent_model = AgentModel(
                self._exp_config,
                self._rl_client
            )

            if self._checkpoint_path is None:
                self.fetch_model()
            else:
                self._agent_model.load_checkpoint(self._checkpoint_path)

    def fetch_model(self):
        if self._checkpoint_path is None:
            self._agent_model.fetch(self._algorithm_id)

    def init_agent_buffers(self):
        buf_capacity = self._exp_config.env.agent_buffer_size
        first_obs = self._env.reset()

        self._agent_buffer = AgentBuffer(
            buf_capacity,
            self._observation_shapes,
            self._observation_dtypes,
            self._action_size
        )
        self._agent_buffer.push_init_observation([first_obs])

    def init_episode_storage(self):
        storage_path = os.path.join(self._logdir, 'episodes')
        create_if_need(storage_path)
        stored_episode_files = len(next(os.walk(storage_path))[2])
        return storage_path, stored_episode_files

    def get_action_repeat_times(self):
        if self._random_repeat_action:
            return random.randint(1, self._repeat_action)
        else:
            return self._repeat_action

    def run(self):

        self.init_agent_buffers()

        if self._store_episodes:
            path_to_episode_storage, stored_episode_index = self.init_episode_storage()
            print('--- stored episodes: {}'.format(stored_episode_index))

        n_steps = 0
        episode_index = 0

        action = None
        env_action = None
        action_repeated = 0
        action_repeat_times = self.get_action_repeat_times()

        # exploration parameters for gradient exploration
        # explore_start_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_end_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_episodes = 500
        # explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
        # explore_temp = explore_start_temp
        
        def prepare_state(state):
            return [np.expand_dims(state, axis=0)]

        while True:

            if action is None or action_repeated == action_repeat_times:

                action_repeated = 0
                action_repeat_times = self.get_action_repeat_times()

                # obtain current state from the buffer
                state = self._agent_buffer.get_current_state(
                    history_len=self._history_len
                )[0]

                # Bernoulli exploration
                # action = np.array(self._agent_model.act_batch(prepare_state(state))[0])
                # env_action = (action + 1.) / 2.
                # env_action = np.clip(env_action, 0., 1.)
                # env_action = np.random.binomial([1]*self._env.action_size, env_action).astype(np.float32)

                if hasattr(self._exploration, 'built_in_algo') and self._exploration.built_in_algo:
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

            env_action_scores = env_action
            env_action = self._action_postprocess(env_action)
            if self._id == 2:
                print(self._id, action, env_action, env_action_scores)
            next_obs, reward, done, info = self._env.step(env_action)

            if reward != 0.:
                print('r', reward)

            action_repeated += 1
            transition = [[next_obs], action, reward, done]
            self._agent_buffer.push_transition(transition)

            n_steps += 1

            if done or (self._step_limit > 0 and n_steps > self._step_limit):

                self._logger.log(episode_index, n_steps)
                episode = self._agent_buffer.get_complete_episode()
                if self._checkpoint_path is None:
                    self._rl_client.store_episode(episode)

                self.fetch_model()

                # save episode on disk
                if self._store_episodes:
                    ep_path = os.path.join(path_to_episode_storage, 'episode_' + str(stored_episode_index)+'.pkl')
                    with open(ep_path, 'wb') as f:
                        pickle.dump(
                        [
                            [obs_part.tolist() for obs_part in episode[0]],
                            episode[1].tolist(),
                            episode[2].tolist(),
                            episode[3].tolist()
                        ], f, pickle.HIGHEST_PROTOCOL)
                    stored_episode_index += 1

                episode_index += 1
                self.init_agent_buffers()
                n_steps = 0
