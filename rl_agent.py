import sys

import os
import random
import pickle
import numpy as np

from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from misc.defaults import default_parse_fn, set_agent_seed, parse_agent_args, init_episode_storage
from misc.rl_logger import RLLogger

class RLAgent:
    
    def __init__(self, env, seed, use_tensorflow=True):
        self._env = env
        self._seed = seed
        self._use_tensorflow = use_tensorflow

    def run(self):
        set_agent_seed(self._seed)
        args, hparams = parse_agent_args()
        logger = RLLogger(args, self._env)

        rl_client = RLClient(
            port=hparams["server"]["init_port"] + args.id)

        if self._use_tensorflow:
            from rl_server.tensorflow.agent_model_tf import AgentModel
            agent_model = AgentModel(hparams, rl_client)
            agent_model.fetch()

        history_len = hparams["server"]["history_length"]
        buf_capacity = hparams["env"]["agent_buffer_size"]

        agent_buffer = AgentBuffer(
            buf_capacity,
            self._env.observation_shapes,
            self._env.action_size)
        agent_buffer.push_init_observation([self._env.reset()])

        if args.store_episodes:
            path_to_episode_storage, stored_episode_index = init_episode_storage(args.id, args.logdir)
            print('stored episodes: {}'.format(stored_episode_index))

        ################################ run agent ################################
        n_steps = 0
        episode_index = 0

        # exploration parameters for gradient exploration
        explore_start_temp = hparams["env"]["ge_temperature"]
        explore_end_temp = hparams["env"]["ge_temperature"]
        explore_episodes = 500
        explore_dt = (explore_start_temp - explore_end_temp) / explore_episodes
        explore_temp = explore_start_temp
        
        def prepare_state(state):
            return [np.expand_dims(state, axis=0)]

        while True:

            # obtain current state from the buffer
            state = agent_buffer.get_current_state(
                history_len=history_len)[0]

            if args.validation:
                # result = rl_client.act_batch([state.ravel()], mode="with_gradients")
                action = agent_model.act_batch(prepare_state(state))[0]
            else:
                if np.float(args.exploration) >= 0:
                    # action = rl_client.act([state.ravel()]) + np.random.normal(
                    #     scale=np.float(args.exploration), size=self._env.action_size)
                    action = agent_model.act_batch(prepare_state(state))[0]
                    action = np.array(action) + np.random.normal(
                        scale=np.float(args.exploration),
                        size=self._env.action_size)
                else:
                    # gradient exploration
                    # result = rl_client.act_batch([state.ravel()], mode="with_gradients")
                    # action = result[0][0]
                    # grad = result[1][0]
                    actions, grads = agent_model.act_batch(
                        prepare_state(state),
                        mode="with_gradients")
                    action = actions[0]
                    grad = grads[0]

                    action = np.array(action)
                    random_action = self._env.get_random_action()
                    explore = 1. - np.clip(
                        np.abs(grad), 0., explore_temp) / explore_temp
                    action = (1 - explore) * action + explore * random_action
                    
            # clip action to be in range [-1, 1]
            action = np.clip(action, -1., 1.)

            next_obs, reward, done, info = self._env.step(action)
            transition = [[next_obs], action, reward, done]
            agent_buffer.push_transition(transition)
            next_state = agent_buffer.get_current_state(
                history_len=history_len)[0].ravel()
            n_steps += 1

            if done:

                logger.log(episode_index, n_steps)
                episode = agent_buffer.get_complete_episode()
                rl_client.store_episode(episode)
                agent_model.fetch()

                # save episode on disk
                if args.store_episodes:
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
