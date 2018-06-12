# DDPG taken from
# https://github.com/nivwusquorum/tensorflow-deepq

import tensorflow as tf


class DDPG(object):
    def __init__(self,
                 actor,
                 critic,
                 target_actor,
                 target_critic,
                 actor_optimizer,
                 critic_optimizer,
                 discount_rate=0.95,
                 target_actor_update_rate=0.01,
                 target_critic_update_rate=0.01,
                 rewards=None,
                 given_action=None,
                 observation_for_act=None,
                 observation=None,
                 next_observation=None,
                 terminator=None):
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
        self._actor = actor
        self._critic = critic
        self._target_actor = target_actor
        self._target_critic = target_critic
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._discount_rate = tf.constant(discount_rate)
        self._target_actor_update_rate = \
            tf.constant(target_actor_update_rate)
        self._target_critic_update_rate = \
            tf.constant(target_critic_update_rate)

        self._rewards = rewards
        self._given_action = given_action
        self._observation_for_act = observation_for_act
        self._observation = observation
        self._next_observation = next_observation
        self._terminator = terminator

        self.create_variables()

    @staticmethod
    def update_target_network(source_network, target_network, update_rate):
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

    @staticmethod
    def assign_network(source_network, target_network):
        return tf.group(*(
            v_target.assign(v_source)
            for v_source, v_target in zip(
                source_network.variables(),
                target_network.variables()
            )
        ))

    def create_variables(self):
        # self._target_actor = self._actor.copy(scope="target_actor")

        # # we need second GPU for inference of actions.
        # # Such setup significantly increases learning by
        # # eliminating competition (for GPU) between train ops
        # # and inference and thus increasing simulation speed
        # # which let get more training data from simulator
        # with tf.device('/device:GPU:1'):
        #     self._actor_gpu_1 = self._actor.copy(scope="actor_gpu_1")
        # self._target_critic = self._critic.copy(scope="target_critic")

        with tf.name_scope("taking_action"):
            self._actor_action = self._actor(
                self._observation_for_act)

        with tf.name_scope("estimating_future_reward"):
            self._next_action = tf.stop_gradient(
                self._target_actor(self._next_observation))

            self._next_value = tf.stop_gradient(
                tf.reshape(
                    self._target_critic(
                        [self._next_observation, self._next_action]),
                    [-1]))

            self._future_reward = self._rewards + self._discount_rate * \
                (1 - self._terminator) * \
                self._next_value

        with tf.name_scope("critic_update"):
            self._value_given_action = tf.reshape(
                self._critic([self._observation, self._given_action]),
                [-1])

            self._critic_error = tf.identity(
                tf.losses.huber_loss(
                    self._value_given_action,
                    self._future_reward),
                name='critic_error')

            critic_gradients = self._critic_optimizer.compute_gradients(
                self._critic_error,
                var_list=self._critic.variables())

            self._critic_update = self._critic_optimizer.apply_gradients(
                critic_gradients,
                name='critic_train_op')

        with tf.name_scope("actor_update"):
            self._actor_score = self._critic(
                [self._observation, self._actor(self._observation)])

            actor_gradients = self._actor_optimizer.compute_gradients(
                tf.reduce_mean(-self._actor_score),
                var_list=self._actor.variables())

            self._actor_update = self._actor_optimizer.apply_gradients(
                actor_gradients,
                name='actor_train_op')

        with tf.name_scope("target_network_update"):
            self._target_actor_update = DDPG.update_target_network(
                self._actor,
                self._target_actor,
                self._target_actor_update_rate)
            self._target_critic_update = DDPG.update_target_network(
                self._critic,
                self._target_critic,
                self._target_critic_update_rate)
            self._update_all_targets = tf.group(
                self._target_actor_update,
                self._target_critic_update,
                name='target_networks_update')

    def get_actor_train_op(self):
        return self._actor_update

    def get_critic_train_op(self):
        return self._critic_update

    def get_loss_op(self):
        return self._critic_error

    def get_update_target_networks_op(self):
        return self._update_all_targets

    def get_actor_action_op(self):
        return self._actor_action
