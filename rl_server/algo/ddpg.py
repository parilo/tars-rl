# DDPG taken from
# https://github.com/nivwusquorum/tensorflow-deepq

import tensorflow as tf


class DDPG(object):
    def __init__(self,
                 observation_shapes,
                 action_size,
                 actor,
                 critic,
                 actor_optimizer,
                 critic_optimizer,
                 discount_rate=0.95,
                 target_actor_update_rate=1.0,
                 target_critic_update_rate=1.0):
        """Initialize the DDPG object.
        Based on:
            https://arxiv.org/abs/1509.02971
        Parameters
        -------
        actor: model
            neural network that implements activate function
            that can take in observation vector or a batch
            and returns action for each observation.
            input shape:  [[batch_size, observation_size], ...]
            output shape: [batch_size, action_size]
        critic: model
            neural network that takes observations and actions
            and returns Q values (scores)
            for each pair of observation and action
            input shape:
            [[[batch_size, observation_size], ...], [batch_size, action_size]]
            output shape: [batch_size, 1]
        optimizer:
            optimizer for train actor and critic
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        target_actor_update_rate: float
            how much to update target critic after each
            iteration. Let's call target_critic_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        target_critic_update_rate: float
            analogous to target_actor_update_rate, but for
            target_critic
        """
        self._observation_shapes = observation_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic = critic
        self._target_actor = actor.copy(scope='target_actor')
        self._target_critic = critic.copy(scope='target_critic')
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._gamma = tf.constant(discount_rate)
        self._target_actor_update_rate = tf.constant(target_actor_update_rate)
        self._target_critic_update_rate = tf.constant(target_critic_update_rate)

        self._create_placeholders()
        self._create_variables()

    @staticmethod
    def _update_target_network(source_network, target_network, update_rate):
        return tf.group(*(
            # this is equivalent to target =
            # (1-alpha) * target + alpha * source
            v_target.assign_sub(
                update_rate * (v_target - v_source))
            for v_source, v_target in zip(
                source_network.variables(),
                target_network.variables()
            )
        ))

    def _create_placeholders(self):
        observation_batch_shapes = []
        for s in self._observation_shapes:
            observation_batch_shapes.append(
                tuple([None] + list(s))
            )

        self._rewards = tf.placeholder(
            tf.float32, (None,), name='inp_rewards'
        )
        self._given_action = tf.placeholder(
            tf.float32, (None, self._action_size),
            name='inp_actions'
        )
        self._observation_for_act = []
        self._observation = []
        self._next_observation = []
        for shape in observation_batch_shapes:
            self._observation_for_act.append(
                tf.placeholder(
                    tf.float32,
                    shape,
                    name='inps_observations_for_act'
                )
            )
            self._observation.append(
                tf.placeholder(tf.float32, shape, name='inp_prev_state')
            )
            self._next_observation.append(
                tf.placeholder(tf.float32, shape, name='inp_next_states')
            )
        self._terminator = tf.placeholder(
            tf.float32, (None,), name='inp_terminator'
        )

    def _create_variables(self):

        with tf.name_scope("taking_action"):
            self._actor_action = self._actor(self._observation_for_act)

        with tf.name_scope("estimating_bellman_equation_sides"):

            # left hand side of the Bellman equation
            self._value_lhs = tf.reshape(self._critic([self._observation, self._given_action]), [-1])

            # right hand side of the Bellman equation
            self._next_action = tf.stop_gradient(self._target_actor(self._next_observation))
            self._next_value = tf.stop_gradient(tf.reshape(self._target_critic(
                [self._next_observation, self._next_action]), [-1]))
            self._value_rhs = self._rewards + self._gamma * (1 - self._terminator) * self._next_value

        with tf.name_scope("critic_update"):

            self._critic_error = tf.losses.huber_loss(self._value_lhs,self._value_rhs)

            critic_gradients = self._critic_optimizer.compute_gradients(
                self._critic_error, var_list=self._critic.variables())

            self._critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        with tf.name_scope("actor_update"):

            self._actor_error = -tf.reduce_mean(self._critic(
                [self._observation, self._actor(self._observation)]))

            actor_gradients = self._actor_optimizer.compute_gradients(
                self._actor_error, var_list=self._actor.variables())

            self._actor_update = self._actor_optimizer.apply_gradients(actor_gradients)

        with tf.name_scope("target_networks_update"):

            self._target_actor_update = DDPG._update_target_network(
                self._actor, self._target_actor, self._target_actor_update_rate)

            self._target_critic_update = DDPG._update_target_network(
                self._critic, self._target_critic, self._target_critic_update_rate)

            self._update_all_targets = tf.group(
                self._target_actor_update, self._target_critic_update)

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def act_batch(self, sess, states):

        feed = {}
        for i in range(len(states)):
            feed[self._observation_for_act[i]] = states[i]

        actions = sess.run(
            self._actor_action,
            feed_dict=feed
        )

        return actions.tolist()

    def train(self, sess, batch):

        feed = {
            self._rewards: batch.r,
            self._given_action: batch.a,
            self._terminator: batch.done
        }
        for i in range(len(batch.s)):
            feed[self._observation[i]] = batch.s[i]
            feed[self._next_observation[i]] = batch.s_[i]

        loss, _, _ = sess.run(
            [
                self._critic_error,
                self._critic_update,
                self._actor_update
            ],
            feed_dict=feed
        )
        return loss

    def target_network_update(self, sess):
        sess.run(self._update_all_targets)
