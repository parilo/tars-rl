import tensorflow as tf
from .base_algo import BaseAlgo


def huber_loss(source, target, weights, kappa=1.0):
    err = tf.subtract(source, target)
    loss = tf.where(
        tf.abs(err) < kappa,
        0.5 * tf.square(err),
        kappa * (tf.abs(err) - 0.5 * kappa))
    return tf.reduce_mean(tf.multiply(loss, weights))


class QuantileDDPG(BaseAlgo):
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            n_step=1,
            actor_grad_val_clip=1.0,
            actor_grad_norm_clip=None,
            critic_grad_val_clip=None,
            critic_grad_norm_clip=None,
            gamma=0.99,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0):
        super(QuantileDDPG, self).__init__(
            state_shapes, action_size, actor, critic,
            actor_optimizer, critic_optimizer, n_step,
            actor_grad_val_clip, actor_grad_norm_clip,
            critic_grad_val_clip, critic_grad_norm_clip,
            gamma, target_actor_update_rate, target_critic_update_rate)

        self.num_atoms = self._critic.num_atoms
        tau_min = 1 / (2 * self.num_atoms)
        tau_max = 1 - tau_min
        self.tau = tf.lin_space(
            start=tau_min, stop=tau_max, num=self.num_atoms)
        self.create_placeholders()
        self.build_graph()

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self._actor(self.states_ph)

        with tf.name_scope("actor_update"):
            atoms = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            q_values = tf.reduce_mean(atoms, axis=-1)
            self.policy_loss = -tf.reduce_mean(q_values)
            self.actor_update = self.get_actor_update(self.policy_loss)

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
                self.tau[None, :, None] - delta_atoms_diff) / self.num_atoms
            self.value_loss = huber_loss(
                atoms[:, :, None], target_atoms[:, None, :], weights)
            self.critic_update = self.get_critic_update(self.value_loss)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_actor_update_op = self.get_target_actor_update()
            self.target_critic_update_op = self.get_target_critic_update()
