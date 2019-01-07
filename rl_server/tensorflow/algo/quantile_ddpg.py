import tensorflow as tf

from .ddpg import DDPG


def huber_loss(source, target, weights, kappa=1.0):
    err = tf.subtract(source, target)
    loss = tf.where(
        tf.abs(err) < kappa,
        0.5 * tf.square(err),
        kappa * (tf.abs(err) - 0.5 * kappa))
    return tf.reduce_mean(tf.multiply(loss, weights))


class QuantileDDPG(DDPG):

    def _get_gradients_wrt_actions(self):
        atoms = self._critic([self.states_ph, self.actions_ph])
        q_values = tf.reduce_mean(atoms, axis=-1)
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def build_graph(self):

        self._num_atoms = self._critic.num_atoms
        tau_min = 1 / (2 * self._num_atoms)
        tau_max = 1 - tau_min
        self._tau = tf.lin_space(
            start=tau_min, stop=tau_max, num=self._num_atoms)

        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            atoms = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            self._q_values = tf.reduce_mean(atoms, axis=-1)
            self._policy_loss = -tf.reduce_mean(self._q_values)
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("critic_update"):
            atoms = self._critic([self.states_ph, self.actions_ph])
            next_actions = self._target_actor(self.next_states_ph)
            next_atoms = self._target_critic(
                [self.next_states_ph, next_actions])
            gamma = self._gamma ** self._n_step
            target_atoms = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * next_atoms
            target_atoms = tf.stop_gradient(target_atoms)
            atoms_diff = target_atoms[:, None, :] - atoms[:, :, None]
            delta_atoms_diff = tf.where(
                atoms_diff < 0,
                tf.ones_like(atoms_diff),
                tf.zeros_like(atoms_diff))
            weights = tf.abs(
                self._tau[None, :, None] - delta_atoms_diff) / self._num_atoms
            self._value_loss = huber_loss(
                atoms[:, :, None], target_atoms[:, None, :], weights)
            self._critic_update = self._get_critic_update(self._value_loss)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_actor_update_op = self._get_target_actor_update()
            self._target_critic_update_op = self._get_target_critic_update()
