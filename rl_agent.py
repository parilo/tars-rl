import sys

import os
import random
import pickle
import numpy as np

from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from misc.defaults import set_agent_seed, init_episode_storage
from misc.rl_logger import RLLogger

np.set_printoptions(suppress=True)

class RLAgent:
    
    def __init__(
        self,
        experiment_config,
        agent_config,
        logdir,
        validation,
        exploration,
        store_episodes,
        agent_id,
        env,
        seed,
        use_tensorflow=True
    ):
        self._exp_config = experiment_config
        self._agent_config = agent_config
        self._logdir = logdir
        self._validation = validation
        self._exploration = exploration
        self._store_episodes = store_episodes
        self._id = agent_id
        self._env = env
        self._seed = seed
        self._use_tensorflow = use_tensorflow
        self._exp_index = 0
        
    def fetch_model(self):
        self._agent_model.fetch(self._id)

    def run(self):
        set_agent_seed(self._seed)
        logger = RLLogger(self._logdir, self._id, self._validation, self._env)

        rl_client = RLClient(
            port=self._exp_config.config["server"]["init_port"] + self._id)

        if self._use_tensorflow:
            from rl_server.tensorflow.agent_model_tf import AgentModel
            self._agent_model = AgentModel(self._exp_config, self._agent_config, rl_client)
            self.fetch_model()

        history_len = self._exp_config.config["env"]["history_length"]
        buf_capacity = self._exp_config.config["env"]["agent_buffer_size"]

        agent_buffer = AgentBuffer(
            buf_capacity,
            self._env.observation_shapes,
            self._env.action_size)
        agent_buffer.push_init_observation([self._env.reset()])

        if self._store_episodes:
            path_to_episode_storage, stored_episode_index = init_episode_storage(self._id, self._logdir)
            print('stored episodes: {}'.format(stored_episode_index))

        ################################ run agent ################################
        n_steps = 0
        episode_index = 0

        # exploration parameters for gradient exploration
        explore_start_temp = self._exp_config.config["env"]["ge_temperature"]
        explore_end_temp = self._exp_config.config["env"]["ge_temperature"]
        explore_episodes = 500
        explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
        explore_temp = explore_start_temp
        
        def prepare_state(state):
            return [np.expand_dims(state, axis=0)]

        while True:

            # obtain current state from the buffer
            state = agent_buffer.get_current_state(
                history_len=history_len)[0]

            # Bernoulli exploration
            # action = np.array(self._agent_model.act_batch(prepare_state(state))[0])
            # env_action = (action + 1.) / 2.
            # env_action = np.clip(env_action, 0., 1.)
            # env_action = np.random.binomial([1]*self._env.action_size, env_action).astype(np.float32)

            if self._validation:
                action = self._agent_model.act_batch(prepare_state(state))[0]
                action = np.array(action)
                env_action = action
            else:
                # exploration with normal noise
                action = self._agent_model.act_batch(prepare_state(state))[0]
                action = np.array(action) + np.random.normal(
                   scale=np.float(self._exploration),
                   size=self._env.action_size)
                env_action = (action + 1.) / 2.
                env_action = np.clip(env_action, 0., 1.)
            
            next_obs, reward, done, info = self._env.step(env_action)
            transition = [[next_obs], action, reward, done]
            agent_buffer.push_transition(transition)
            next_state = agent_buffer.get_current_state(
                history_len=history_len)[0].ravel()
            n_steps += 1

            if done:

                logger.log(episode_index, n_steps)
                episode = agent_buffer.get_complete_episode()
                rl_client.store_episode(episode)
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
                agent_buffer = AgentBuffer(
                    buf_capacity, self._env.observation_shapes, self._env.action_size)
                agent_buffer.push_init_observation([self._env.reset()])
                n_steps = 0
