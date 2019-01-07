import tensorflow as tf

from .td3 import TD3


def huber_loss(source, target, weights, kappa=1.0):
    err = tf.subtract(source, target)
    loss = tf.where(
        tf.abs(err) < kappa,
        0.5 * tf.square(err),
        kappa * (tf.abs(err) - 0.5 * kappa))
    return tf.reduce_mean(tf.multiply(loss, weights))


class QuantileTD3(TD3):

    def build_graph(self):

        self._num_atoms = self._critic1.num_atoms
        tau_min = 1 / (2 * self._num_atoms)
        tau_max = 1 - tau_min
        self._tau = tf.lin_space(
            start=tau_min, stop=tau_max, num=self._num_atoms)

        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            q_values1 = tf.reduce_mean(self._critic1(
                [self.states_ph, self._actor(self.states_ph)]), axis=-1)
            q_values2 = tf.reduce_mean(self._critic2(
                [self.states_ph, self._actor(self.states_ph)]), axis=-1)
            self._q_values_min = tf.minimum(q_values1, q_values2)
            self._policy_loss = -tf.reduce_mean(self._q_values_min)
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("critic_update"):
            atoms1 = self._critic1([self.states_ph, self.actions_ph])
            atoms2 = self._critic2([self.states_ph, self.actions_ph])

            next_actions = self._target_actor(self.next_states_ph)
            actions_noise = tf.random_normal(
                tf.shape(next_actions), mean=0.0, stddev=self._act_noise_std)
            clipped_noise = tf.clip_by_value(
                actions_noise, -self._act_noise_clip, self._act_noise_clip)
            next_actions = next_actions + clipped_noise

            next_atoms1 = self._target_critic1(
                [self.next_states_ph, next_actions])
            next_atoms2 = self._target_critic2(
                [self.next_states_ph, next_actions])
            next_q_values1 = tf.reduce_mean(next_atoms1, axis=-1)
            next_q_values2 = tf.reduce_mean(next_atoms2, axis=-1)
            q_diff = next_q_values1 - next_q_values2
            mask = tf.where(
                q_diff < 0,
                tf.ones_like(q_diff),
                tf.zeros_like(q_diff))[:, None]
            next_atoms = mask * next_atoms1 + (1 - mask) * next_atoms2
            gamma = self._gamma ** self._n_step
            target_atoms = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * next_atoms
            target_atoms = tf.stop_gradient(target_atoms)

            self._value_loss = self.quantile_loss(atoms1, target_atoms)
            self._value_loss2 = self.quantile_loss(atoms2, target_atoms)

            self._critic1_update = self._get_critic1_update(self._value_loss)
            self._critic2_update = self._get_critic2_update(self._value_loss2)
            self._critic_update = tf.group(
                self._critic1_update, self._critic2_update)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_actor_update_op = self._get_target_actor_update()
            self._target_critic_update_op = tf.group(
                self._get_target_critic1_update(),
                self._get_target_critic2_update())

    def quantile_loss(self, atoms, target_atoms):
        atoms_diff = target_atoms[:, None, :] - atoms[:, :, None]
        delta_atoms_diff = tf.where(
            atoms_diff < 0,
            tf.ones_like(atoms_diff),
            tf.zeros_like(atoms_diff))
        weights = tf.abs(
            self._tau[None, :, None] - delta_atoms_diff) / self._num_atoms
        loss = huber_loss(
            atoms[:, :, None], target_atoms[:, None, :], weights)
        return loss
