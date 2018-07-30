#!/usr/bin/env python

import tensorflow as tf
import random
import numpy as np
from rl_server.server.rl_server_api import RLServerAPI
from rl_server.rl_train_loop import RLTrainLoop


class RLServer:

    def __init__(self,
                 num_clients,
                 action_size,
                 observation_shapes,
                 state_shapes,
                 model_load_callback,
                 agent_algorithm,
                 action_dtype=tf.float32,
                 is_actions_space_continuous=True,
                 gpu_id=0,
                 batch_size=256,
                 experience_replay_buffer_size=1000000,
                 use_prioritized_buffer=True,
                 use_synchronous_update=True,
                 train_every_nth=4,
                 history_length=3,
                 start_learning_after=5000,
                 target_critic_update_period=500,
                 target_actor_update_period=500,
                 show_stats_period=20,
                 save_model_period=10000,
                 init_port=8777,
                 ckpt_path='ckpt/'):

        self._server_api = RLServerAPI(
            num_clients,
            observation_shapes,
            state_shapes,
            init_port=init_port)

        self._train_loop = RLTrainLoop(
            observation_shapes=observation_shapes,
            action_size=action_size,
            action_dtype=action_dtype,
            is_actions_space_continuous=is_actions_space_continuous,
            gpu_id=gpu_id,
            batch_size=batch_size,
            experience_replay_buffer_size=experience_replay_buffer_size,
            use_prioritized_buffer=use_prioritized_buffer,
            use_synchronous_update=use_synchronous_update,
            train_every_nth=train_every_nth,
            history_length=history_length,
            start_learning_after=start_learning_after,
            target_critic_update_period=target_critic_update_period,
            target_actor_update_period=target_actor_update_period,
            show_stats_period=show_stats_period,
            save_model_period=save_model_period,
            ckpt_path=ckpt_path)

        self._train_loop.set_algorithm(agent_algorithm)
        self._train_loop.init_vars(model_load_callback)
        self._server_api.set_act_batch_callback(self._train_loop.act_batch)
        self._server_api.set_act_with_gradient_batch_callback(self._train_loop.act_with_gradient_batch)
        self._server_api.set_store_episode_callback(self._train_loop.store_episode)

    def start(self):
        print('--- starting rl server')
        self._server_api.start_server()
        self._train_loop.start_training()
