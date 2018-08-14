import os
import tensorflow as tf
import time
from rl_server.server.server_replay_buffer import ServerBuffer
from threading import Lock, Thread
import multiprocessing
from tensorboardX import SummaryWriter
from misc.defaults import create_if_need
from datetime import datetime
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


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(num_cpu=1)


class TFRLTrainer(RLTrainer):
    def init(self):
        self._sess = make_session(num_cpu=1, make_default=True)
        self._logger = tf.summary.FileWriter("logs")

        self._logger.add_graph(self._sess.graph)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(max_to_keep=None)
        self._algo.target_network_init(self._sess)

    def load_checkpoint(self, path):
        self._saver.restore(self._sess, path)

    def act_batch(self, states, mode="default"):
        if mode == "default":
            actions = self._algo.act_batch(self._sess, states)
        if mode == "sac_deterministic":
            actions = self._algo.act_batch_deterministic(self._sess, states)
        if mode == "with_gradients":
            # actually, here it will return actions and grads
            actions = self._algo.act_batch_with_gradients(self._sess, states)
        if self._use_synchronous_update:
            self.train_loop_step()
        return actions

    def train_step(self):

        if self._use_prioritized_buffer:

            prio_batch = self.server_buffer.get_prioritized_batch(
                self._batch_size,
                history_len=self._hist_len,
                n_step=self._n_step,
                beta=self._beta,
                gamma=self._gamma)
            batch, indices, is_weights = prio_batch
            loss = self._algo.train(self._sess, batch, is_weights)
            td_errors = self._algo.get_td_errors(self._sess, batch).ravel()
            self.server_buffer.update_td_errors(indices, td_errors)
            self._beta = min(1.0, self._beta + 1e-6)
        else:
            batch = self.server_buffer.get_batch(
                self._batch_size,
                history_len=self._hist_len,
                n_step=self._n_step,
                gamma=self._gamma)
            loss = self._algo.train(self._sess, batch)
            
            # parallel buffer
            # batch = self.server_buffer.get_batch()
            # if batch is not None:
            #     loss = self._algo.train(self._sess, batch)
            # else:
            #     loss = None

        if self._step_index % self._target_critic_update_period == 0:
            self._algo.target_critic_update(self._sess)
        
        if self._step_index % self._target_actor_update_period == 0:
            self._algo.target_actor_update(self._sess)
        
        if self._step_index % self._show_stats_period == 0:
            queue_size = self.server_buffer.get_stored_in_buffer()
            print(
                "trains: {} loss: {} stored: {}".format(
                    self._step_index, loss, queue_size))

        self.save()

    def save(self):
        if self._step_index % self._save_model_period == 0:
            save_path = self._saver.save(
                self._sess, os.path.join(self._logdir, "model-{}.ckpt".format(
                    self._step_index)))
            print("Model saved in file: %s" % save_path)
            self._n_saved += 1
            # self._logger.add_scalar(
            #     "num saved",
            #     self._n_saved,
            #     self._step_index)

    def get_weights(self, index=0):
        return self._algo.get_weights(self._sess, index)
