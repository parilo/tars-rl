import numpy as np
import tempfile
import tensorflow as tf
import threading
from .algo.ddpg import DDPG
from .osim_rl_model_dense import OSimRLModelDense


class OSimRLDDPG ():

    def __init__(
        self,
        observation_shapes,
        action_size,
        discount_rate=0.99,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4),
        target_actor_update_rate=1.0,
        target_critic_update_rate=1.0
    ):

        critic_shapes = list(observation_shapes)
        critic_shapes.append((action_size,))

        self._critic = OSimRLModelDense(
            input_shapes=critic_shapes,
            output_size=1,
            scope='critic'
        )

        self._target_critic = OSimRLModelDense(
            input_shapes=critic_shapes,
            output_size=1,
            scope='target_critic'
        )

        self._actor = OSimRLModelDense(
            input_shapes=observation_shapes,
            output_size=action_size,
            scope='actor'
        )

        self._target_actor = OSimRLModelDense(
            input_shapes=observation_shapes,
            output_size=action_size,
            scope='target_actor'
        )

        for v in self._actor.variables():
            print('--- actor v: {}'.format(v.name))
        for v in self._critic.variables():
            print('--- critic v: {}'.format(v.name))

        observation_batch_shapes = []
        for s in observation_shapes:
            observation_batch_shapes.append(
                tuple([None] + list(s))
            )

        self._inp_rewards = tf.placeholder(
            tf.float32, (None,), name='inp_rewards'
        )
        self._inp_actions = tf.placeholder(
            tf.float32, (None, action_size),
            name='inp_actions'
        )
        self._inps_observations_for_act = []
        self._inps_prev_states = []
        self._inps_next_states = []
        for shape in observation_batch_shapes:
            self._inps_observations_for_act.append(
                tf.placeholder(
                    tf.float32,
                    shape,
                    name='inps_observations_for_act'
                )
            )
            self._inps_prev_states.append(
                tf.placeholder(tf.float32, shape, name='inp_prev_state')
            )
            self._inps_next_states.append(
                tf.placeholder(tf.float32, shape, name='inp_next_states')
            )
        self._inp_terminator = tf.placeholder(
            tf.float32, (None,), name='inp_terminator'
        )

        self._controller = DDPG(
            actor=self._actor,
            critic=self._critic,
            target_actor=self._target_actor,
            target_critic=self._target_critic,
            actor_optimizer=optimizer,
            critic_optimizer=optimizer,
            discount_rate=discount_rate,
            target_actor_update_rate=target_actor_update_rate,
            target_critic_update_rate=target_critic_update_rate,
            rewards=self._inp_rewards,
            given_action=self._inp_actions,
            observation_for_act=self._inps_observations_for_act,
            observation=self._inps_prev_states,
            next_observation=self._inps_next_states,
            terminator=self._inp_terminator
        )

        self._actor_train_op = self._controller.get_actor_train_op()
        self._critic_train_op = self._controller.get_critic_train_op()
        self._loss_op = self._controller.get_loss_op()
        self._update_target_networks_op = \
            self._controller.get_update_target_networks_op()
        self._actor_action_op = self._controller.get_actor_action_op()

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def act_batch(self, sess, states):

        feed = {}
        for i in range(len(states)):
            feed[self._inps_observations_for_act[i]] = states[i]

        actions = sess.run(
            self._actor_action_op,
            feed_dict=feed
        )

        return actions.tolist()

    def train(self, sess, batch):

        feed = {
            self._inp_rewards: batch.r,
            self._inp_actions: batch.a,
            self._inp_terminator: batch.done
        }
        for i in range(len(batch.s)):
            feed[self._inps_prev_states[i]] = batch.s[i]
            feed[self._inps_next_states[i]] = batch.s_[i]

        loss, _, _ = sess.run(
            [
                self._loss_op,
                self._critic_train_op,
                self._actor_train_op
            ],
            feed_dict=feed
        )
        return loss

    def target_network_update(self, sess):
        sess.run(self._update_target_networks_op)
