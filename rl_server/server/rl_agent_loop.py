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

            low_from = np.array(low_from) if isinstance(low_from, list) else low_from
            low_to = np.array(low_to) if isinstance(low_to, list) else low_to
            high_from = np.array(high_from) if isinstance(high_from, list) else high_from
            high_to = np.array(high_to) if isinstance(high_to, list) else high_to

            def remap_func (action):
                return (action - low_from) / (high_from - low_from) * (high_to - low_to) + low_to
            self._action_remap_function = remap_func

        # action clipping function
        def trivial_clipping(action):
            return action
        self._clipping_function = trivial_clipping
        if exp_config.isset('actor'):
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

                expl_type_set = (
                    self._exploration is not None and
                    self._exploration.isset('type')
                )

                boltzmann_expl = expl_type_set and self._exploration.type == 'boltzmann'
                e_greedy_expl = expl_type_set and self._exploration.type == 'e-greedy'

                if boltzmann_expl:
                    possible_actions = list(range(0, self._exp_config.env.action_size))
                    boltzmann_expl_temp = self._exploration.temp

                if e_greedy_expl:
                    random_prob = self._exploration.random_prob

                def softmax_action_postprocess(action):
                    if boltzmann_expl:
                        # norm = np.linalg.norm(action)
                        self._action_scores = softmax(action / boltzmann_expl_temp)
                        return np.random.choice(possible_actions, p=self._action_scores)
                    if e_greedy_expl:
                        self._action_scores = action
                        if random.random() < random_prob:
                            return random.randint(0, self._exp_config.env.action_size - 1)
                        else:
                            return np.argmax(softmax(action))
                    else:
                        return np.argmax(softmax(action))

                self._action_postprocess = softmax_action_postprocess

        # action remap
        self._action_remap_discrete_func = None
        if exp_config.env.isset('action_remap_discrete'):
            action_discrete_mapping = exp_config.as_obj()['env']['action_remap_discrete']
            def action_remap_discrete_func(action):
                return action_discrete_mapping[action]
            self._action_remap_discrete_func = action_remap_discrete_func

        self._discrete_actions = False
        if exp_config.env.isset('discrete_actions'):
            self._discrete_actions = exp_config.env.discrete_actions

        # store episodes
        self._store_episodes = False
        if hasattr(agent_config, 'store_episodes') and agent_config.store_episodes:
            self._store_episodes = True

        # reward clipping
        self._reward_clip_max = None
        self._reward_clip_min = None
        if exp_config.env.isset('reward_clip'):
            if exp_config.env.reward_clip.isset('max'):
                self._reward_clip_max = exp_config.env.reward_clip.max
            if exp_config.env.reward_clip.isset('min'):
                self._reward_clip_min = exp_config.env.reward_clip.min

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
            self._action_size,
            self._discrete_actions
        )
        if isinstance(first_obs, list):
            self._agent_buffer.push_init_observation(first_obs)
        else:
            self._agent_buffer.push_init_observation([first_obs])

    def init_episode_storage(self):
        storage_path = os.path.join(self._logdir, 'episodes_' + str(self._id))
        create_if_need(storage_path)
        stored_episode_files = len(next(os.walk(storage_path))[2])
        return storage_path, stored_episode_files

    def get_action_repeat_times(self):
        if self._random_repeat_action:
            return random.randint(1, self._repeat_action)
        else:
            return self._repeat_action

    def is_states_the_same(self, state1, state2):
        for s1, s2 in zip(state1, state2):
            if np.any(s1 != s2):
                return False
        return True

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

        prev_state = None
        same_state_repeat = 0

        # exploration parameters for gradient exploration
        # explore_start_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_end_temp = self._exp_config.config["env"]["ge_temperature"]
        # explore_episodes = 500
        # explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
        # explore_temp = explore_start_temp

        # avg_v = 0.
        # avg_v_2 = 0.
        # avg_v_discout = 0.02

        # prev_v = 0.
        # prev_v2 = 0.
        
        def prepare_state(state):
            if isinstance(state, list):
                return [np.expand_dims(s_part, axis=0) for s_part in state]
            else:
                return [np.expand_dims(state, axis=0)]

        while True:

            if action is None or action_repeated == action_repeat_times:

                action_repeated = 0
                action_repeat_times = self.get_action_repeat_times()

                # obtain current state from the buffer
                state = self._agent_buffer.get_current_state(
                    history_len=self._history_len
                )

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
                    # action_target = self._agent_model.act_batch_target(prepare_state(state))[0]
                    action = np.array(action)
                    # action_target = np.array(action_target)
                    # if self._id ==  2: print('--- action before expl', action)

                    # if self._id ==  2: print('--- expl')
                    if not self._validation:
                        # if self._id == 2: print('--- expl expl')
                        if self._exploration.normal_noise is not None:
                            # exploration with normal noise
                            action += np.random.normal(
                               scale=self._exploration.normal_noise,
                               size=self._action_size
                            )
                            # if self._id == 2: print('--- expl normal noise', action)
                        if self._exploration.random_action_prob is not None:
                            # if self._id == 2: print('--- expl random action')
                            if random.random() < self._exploration.random_action_prob:
                                action = self._env.get_random_action()
                                # if self._id == 2: print('--- expl random action 2', action)

                    action = self._clipping_function(action)
                    # if self._id ==  2: print('--- action after expl', action)

                # action remap function
                if self._action_remap_function is not None:
                    env_action = self._action_remap_function(action)
                else:
                    env_action = action

            env_action = self._action_postprocess(env_action)

            # if self._id == 1 and n_steps % 100 == 0:
            #     print(self._id, env_action)

            # if not self._validation and n_steps % 500 == 0:
            #     if hasattr(self, '_action_scores'):
            #         print(self._id, '\n', env_action_scores, '\n', self._action_scores)

            if self._action_remap_discrete_func is not None:
                env_action_remapped = self._action_remap_discrete_func(env_action)
            else:
                env_action_remapped = env_action

            # if self._id == 2:
            #     print('--- action', env_action_remapped)

            next_obs, reward, done, info = self._env.step(env_action_remapped)

            if self._reward_clip_max:
                reward = min(reward, self._reward_clip_max)

            if self._reward_clip_min and reward > 0.:
                reward = max(reward, self._reward_clip_min)

            # if reward > 0.:
            #     print('r', reward)

            # np.save('obs-{}-{}.npy'.format(episode_index, n_steps), next_obs)

            # if reward != 0.:
            #     print('r', reward)

            action_repeated += 1

            if self._discrete_actions:
                action_to_save = env_action
            else:
                action_to_save = action

            if not isinstance(next_obs, list):
                next_obs = [next_obs]

            # reward += dv2
            # print('r', reward)
            transition = [next_obs, action_to_save, reward, done]
            self._agent_buffer.push_transition(transition)

            n_steps += 1

            # print(prev_action, same_action_repeats)
            if self._exp_config.env.isset('stop_if_same_state_repeat'):
                if prev_state is not None and self.is_states_the_same(prev_state, state):
                    same_state_repeat += 1
                    if same_state_repeat > self._exp_config.env.stop_if_same_state_repeat:
                        done = True
                        prev_state = None
                        same_state_repeat = 0
                else:
                    prev_state = state
                    same_state_repeat = 0

            if done or (self._step_limit > 0 and n_steps > self._step_limit):

                self._agent_model.reset_states()

                self._logger.log_dict(self._env.get_logs(), episode_index)
                self._logger.log(episode_index, n_steps)
                episode = self._agent_buffer.get_complete_episode()
                if self._checkpoint_path is None:
                    self._rl_client.store_episode(episode)

                self.fetch_model()

                # save episode on disk
                if self._store_episodes:
                    ep_path = os.path.join(path_to_episode_storage, 'episode_' + str(stored_episode_index)+'.pkl')
                    with open(ep_path, 'wb') as f:
                        pickle.dump(episode, f, pickle.HIGHEST_PROTOCOL)
                        # pickle.dump(
                        # [
                        #     [obs_part.tolist() for obs_part in episode[0]],
                        #     episode[1].tolist(),
                        #     episode[2].tolist(),
                        #     episode[3].tolist()
                        # ], f, pickle.HIGHEST_PROTOCOL)
                    stored_episode_index += 1

                episode_index += 1
                self.init_agent_buffers()
                n_steps = 0
