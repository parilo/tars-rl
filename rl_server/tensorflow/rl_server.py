from rl_server.server.rl_server_api import RLServerAPI
from rl_server.server.rl_trainer_tf import TFRLTrainer as RLTrainer


class RLServer:

    def __init__(self, exp_config, agent_algorithm):
        (
            observation_shapes,
            observation_dtypes,
            state_shapes,
            action_size
        ) = exp_config.get_env_shapes()

        self._server_api = RLServerAPI(
            exp_config.server.num_clients,
            observation_shapes,
            state_shapes,
            init_port=exp_config.server.client_start_port)

        self._train_loop = RLTrainer(
            observation_shapes=observation_shapes,
            observation_dtypes=observation_dtypes,
            action_size=action_size,
            experience_replay_buffer_size=exp_config.server.experience_replay_buffer_size,
            use_prioritized_buffer=exp_config.server.use_prioritized_buffer,
            n_step=exp_config.algorithm.n_step,
            gamma=exp_config.algorithm.gamma,
            train_every_nth=exp_config.server.train_every_nth,
            history_length=exp_config.env.history_length,
            start_learning_after=exp_config.server.start_learning_after,
            target_critic_update_period=exp_config.server.target_critic_update_period,
            target_actor_update_period=exp_config.server.target_actor_update_period,
            show_stats_period=exp_config.server.show_stats_period,
            save_model_period=exp_config.server.save_model_period,
            logdir=exp_config.server.logdir)

        self._train_loop.set_algorithm(agent_algorithm)
        self._train_loop.init()
        self._server_api.set_store_episode_callback(self._train_loop.store_episode)
        self._server_api.set_get_weights_callback(self._train_loop.get_weights)

        exp_config.store(exp_config.server.logdir)
        
    def load_weights(self, path):
        self._train_loop.load_checkpoint(path)

    def start(self):
        print("--- starting rl server")
        self._server_api.start_server()
        self._train_loop.start_training()
