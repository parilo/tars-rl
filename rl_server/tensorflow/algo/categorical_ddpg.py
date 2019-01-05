import tensorflow as tf

from .ddpg import DDPG


class CategoricalDDPG(DDPG):

    def _get_gradients_wrt_actions(self):
        logits = self._critic([self.states_ph, self.actions_ph])
        probs = tf.nn.softmax(logits)
        q_values = tf.reduce_sum(probs * self._z, axis=-1)
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def build_graph(self):
        self._num_atoms = self._critic.num_atoms
        self._v_min, self._v_max = self._critic.v
        self._delta_z = (self._v_max - self._v_min) / (self._num_atoms - 1)
        self._z = tf.lin_space(
            start=self._v_min, stop=self._v_max, num=self._num_atoms)
        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            logits = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            probs = tf.nn.softmax(logits)
            q_values = tf.reduce_sum(probs * self._z, axis=-1)
            self._policy_loss = -tf.reduce_mean(q_values)
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("critic_update"):
            logits = self._critic([self.states_ph, self.actions_ph])
            next_actions = self._target_actor(self.next_states_ph)
            next_logits = self._target_critic(
                [self.next_states_ph, next_actions])
            next_probs = tf.nn.softmax(next_logits)
            gamma = self._gamma ** self._n_step
            target_atoms = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * self._z
            tz = tf.clip_by_value(target_atoms, self._v_min, self._v_max)
            tz_z = tz[:, None, :] - self._z[None, :, None]
            tz_z = tf.clip_by_value(
                (1.0 - (tf.abs(tz_z) / self._delta_z)), 0., 1.)
            target_probs = tf.einsum("bij,bj->bi", tz_z, next_probs)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=tf.stop_gradient(target_probs))
            self._value_loss = tf.reduce_mean(cross_entropy)
            self._critic_update = self._get_critic_update(self._value_loss)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_actor_update_op = self._get_target_actor_update()
            self._target_critic_update_op = self._get_target_critic_update()
