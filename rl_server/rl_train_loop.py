import os
import numpy as np
import tempfile
import tensorflow as tf
import time
from .experience_replay_buffer import ExperienceReplayBuffer
from threading import Lock


class RLTrainLoop():

    def __init__(
        self,
        observation_shapes,
        action_size,
        action_dtype,
        is_actions_space_continuous,
        gpu_id=0,
        batch_size=96,
        experience_replay_buffer_size=1000000,
        store_every_nth=4,
        train_every_nth=4,
        start_learning_after=5000,
        target_networks_update_period=500,  # of train ops invokes
        show_stats_period=2000,
        save_model_period=10000
    ):
        self._observation_shapes = observation_shapes
        self._action_size = action_size
        self._action_dtype = action_dtype
        self._batch_size = batch_size
        self._experience_replay_buffer_size = experience_replay_buffer_size
        self._start_learning_after = start_learning_after
        self._store_every_nth = store_every_nth
        self._train_every_nth = train_every_nth
        self._target_networks_update_period = target_networks_update_period
        self._show_stats_period = show_stats_period
        self._save_model_period = save_model_period

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads= 1
        self._sess = tf.Session(config=config)
        
        self._logger = tf.summary.FileWriter("logs")

        self._exp_replay_buffer = ExperienceReplayBuffer(
            self._experience_replay_buffer_size,
            self._store_every_nth,
            num_of_parts_in_state=len(observation_shapes)
        )

        self._store_lock = Lock()
        self._store_index = 0

    def get_tf_session(self):
        return self._sess

    def get_action_size(self):
        return self._action_size

    def get_observation_shapes(self):
        return self._observation_shapes

    def get_observations_for_act_placeholders(self):
        return self._exp_replay_buffer.get_observations_for_act_placeholders()

    def get_train_batches(self):
        return self._exp_replay_buffer.get_train_batches()

    def init_vars(self, model_load_callback=None):
        self._logger.add_graph(self._sess.graph)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(max_to_keep=None)
        if model_load_callback is not None:
            model_load_callback(self._sess, self._saver)

    def act_batch(self, states):
        return self._algo.act_batch(self._sess, states)

    def store_exp_batch(
        self,
        rewards,
        actions,
        prev_states,
        next_states,
        terminators
    ):
        with self._store_lock:
            self._exp_replay_buffer.store_exp_batch(
                rewards,
                actions,
                prev_states,
                next_states,
                terminators
            )

            buffer_size = self._exp_replay_buffer.get_buffer_size()
            if (
                self._store_index % self._train_every_nth == 0 and
                buffer_size > self._start_learning_after
            ):
                self.train_step(self._store_index)
            elif (
                buffer_size < self._start_learning_after and
                self._store_index % 10 == 0
            ):
                print('--- buffer size {}'.format(
                    buffer_size
                ))

            self._store_index += 1

    def set_algorithm(self, algo):
        self._algo = algo

    def train_step(self, step_index):

        batch = self._exp_replay_buffer.sample(self._batch_size)
        queue_size = self._exp_replay_buffer.get_stored_count()
        loss = self._algo.train(self._sess, batch)

        if step_index % self._target_networks_update_period == 0:
            print('--- target network update')
            self._algo.target_network_update(self._sess)

        if step_index % self._show_stats_period == 0:
            print(
                ('trains: {} rewards: {} loss: {}' +
                 ' stored: {} buffer size {}').format(
                    step_index,
                    self._exp_replay_buffer.get_sum_rewards(),
                    loss,
                    queue_size,
                    self._exp_replay_buffer.get_buffer_size()
                )
            )
            self._exp_replay_buffer.reset_sum_rewards()

        if step_index % self._save_model_period == 0:
            save_path = self._saver.save(
                self._sess,
                'ckpt/model-{}.ckpt'.format(step_index)
            )
            print("Model saved in file: %s" % save_path)
