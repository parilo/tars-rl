from rl_server.server.rl_trainer import RLTrainer


class TorchRLTrainer(RLTrainer):
    def init(self):
        self._algo.target_network_init()

    def load_checkpoint(self, load_info):
        self._algo.load(load_info.dir, load_info.index)

    def _train_on_batch(self):
        batch = self.get_batch()

        if self._use_prioritized_buffer:

            batch, indices, is_weights = batch
            train_info = self._algo.train(self._step_index, batch, is_weights)
            td_errors = self._algo.get_td_errors(batch).ravel()
            self.server_buffer.update_td_errors(indices, td_errors)
            self._beta = min(1.0, self._beta + 1e-6)

        else:

            train_info = self._algo.train(self._step_index, batch)

        return train_info

    def _train_on_episodes(self):
        batch_size = self._algo.get_batch_size(self._step_index)
        batch_of_episodes = self.server_buffer.get_batch(batch_size)
        return self._algo.train(self._step_index, batch_of_episodes)

    def train_step(self):
        queue_size = self.server_buffer.get_stored_in_buffer()
        self._logger.log_buffer_size(queue_size, self._step_index)

        if self._algo.is_trains_on_episodes():
            train_info = self._train_on_episodes()
        else:
            train_info = self._train_on_batch()

        if self._algo.is_on_policy():
            self.server_buffer.clear()

        self._logger.log_train(train_info, self._step_index)

        if self._step_index % self._target_critic_update_period == 0:
            self._algo.target_critic_update()

        if self._step_index % self._target_actor_update_period == 0:
            self._algo.target_actor_update()

        if self._step_index % self._show_stats_period == 0:
            batch_size = self._algo.get_batch_size(self._step_index)
            print(
                "step: {} {} train: {} stored: {}".format(
                    self._step_index,
                    batch_size,
                    train_info,
                    self.server_buffer.get_stored_in_buffer_info()
                )
            )

        self._save()

    def _save(self):
        if self._step_index % self._save_model_period == 0:
            self.save(self._logdir, self._step_index)

    def save(self, dir, index):
        self._algo.save(dir, index)
        self._n_saved += 1

    def get_weights(self, algo_index=0):
        """
        :param algo_index: index of the algo in the ensemble
        :return:
        """
        return self._algo.get_weights(algo_index)
