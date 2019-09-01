import os
import multiprocessing
import time

import tensorflow as tf

from rl_server.server.rl_trainer import RLTrainer


def make_session(num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU"s only"""
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


class TFRLTrainer(RLTrainer):
    def init(self):
        self._sess = make_session(num_cpu=1, make_default=True)
        self._saver = tf.train.Saver(max_to_keep=None)
        self._algo.init(self._sess)
        self._algo.target_network_init(self._sess)

    def load_checkpoint(self, path):
        self._saver.restore(self._sess, path)

    def train_step(self):

        queue_size = self.server_buffer.get_stored_in_buffer()
        self._logger.log_buffer_size(queue_size, self._step_index)

        if self._use_prioritized_buffer:

            batch_size = self._algo.get_batch_size(self._step_index)
            prio_batch = self.server_buffer.get_prioritized_batch(
                batch_size,
                history_len=self._hist_len,
                n_step=self._n_step,
                beta=self._beta,
                gamma=self._gamma)
            batch, indices, is_weights = prio_batch
            train_info = self._algo.train(
                sess=self._sess,
                step_index=self._step_index,
                batch=batch,
                is_weights=is_weights
            )
            td_errors = self._algo.get_td_errors(self._sess, batch).ravel()
            self.server_buffer.update_td_errors(indices, td_errors)
            self._beta = min(1.0, self._beta + 1e-6)
        else:
            batch_size = self._algo.get_batch_size(self._step_index)
            batch = self.server_buffer.get_batch(
                batch_size,
                history_len=self._hist_len,
                n_step=self._n_step,
                gamma=self._gamma)
            train_info = self._algo.train(self._sess, self._step_index, batch)

        self._logger.log_train(train_info, self._step_index)

        if self._step_index % self._target_critic_update_period == 0:
            self._algo.target_critic_update(self._sess)

        if self._target_actor_update_period is not None:
            if self._step_index % self._target_actor_update_period == 0:
                self._algo.target_actor_update(self._sess)
        
        if self._step_index % self._show_stats_period == 0:
            cur_time = time.time()
            print(
                "step: {} {} train: {} stored: {} time: {}".format(
                    self._step_index,
                    batch_size,
                    train_info,
                    queue_size,
                    cur_time - self._start_train_time
                )
            )
            self._start_train_time = cur_time

        self.save()

    def save(self):
        if self._step_index % self._save_model_period == 0:
            save_path = self._saver._save(
                self._sess, os.path.join(self._logdir, "model-{}.ckpt".format(
                    self._step_index)))
            print("Model saved in file: %s" % save_path)
            self._n_saved += 1

    def get_weights(self, index=0):
        return self._algo.get_weights(self._sess, index)
