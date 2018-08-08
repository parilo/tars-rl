#!/usr/bin/env python

from rl_server.server.rl_server_api import RLServerAPI
from rl_server.server.rl_trainer_tf import TFRLTrainer as RLTrainer


class RLServer:

    def __init__(self,
                 num_clients,
                 action_size,
                 observation_shapes,
                 state_shapes,
                 agent_algorithm,
                 batch_size=256,
                 experience_replay_buffer_size=1000000,
                 use_prioritized_buffer=True,
                 use_synchronous_update=True,
                 n_step=1,
                 train_every_nth=4,
                 history_length=3,
                 start_learning_after=5000,
                 target_critic_update_period=500,
                 target_actor_update_period=500,
                 show_stats_period=20,
                 save_model_period=10000,
                 init_port=8777,
                 ckpt_path="ckpt/"):

        self._server_api = RLServerAPI(
            num_clients,
            observation_shapes,
            state_shapes,
            init_port=init_port)

        self._train_loop = RLTrainer(
            observation_shapes=observation_shapes,
            action_size=action_size,
            batch_size=batch_size,
            experience_replay_buffer_size=experience_replay_buffer_size,
            use_prioritized_buffer=use_prioritized_buffer,
            use_synchronous_update=use_synchronous_update,
            n_step=n_step,
            train_every_nth=train_every_nth,
            history_length=history_length,
            start_learning_after=start_learning_after,
            target_critic_update_period=target_critic_update_period,
            target_actor_update_period=target_actor_update_period,
            show_stats_period=show_stats_period,
            save_model_period=save_model_period,
            logdir=ckpt_path)

        self._train_loop.set_algorithm(agent_algorithm)
        self._train_loop.init()
        self._server_api.set_act_batch_callback(self._train_loop.act_batch)
        self._server_api.set_store_episode_callback(self._train_loop.store_episode)
        
    def load_weights(self, path):
        self._train_loop.load_checkpoint(path)

    def start(self):
        print("--- starting rl server")
        self._server_api.start_server()
        self._train_loop.start_training()
