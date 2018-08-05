import tensorflow as tf
from .base_algo import BaseAlgo


class CategoricalDDPG(BaseAlgo):
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
        super(CategoricalDDPG, self).__init__(
            state_shapes, action_size, actor, critic,
            actor_optimizer, critic_optimizer, n_step,
            actor_grad_val_clip, actor_grad_norm_clip,
            critic_grad_val_clip, critic_grad_norm_clip,
            gamma, target_actor_update_rate, target_critic_update_rate)

        self.num_atoms = self._critic.num_atoms
        self.v_min, self.v_max = self._critic.v
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z = tf.lin_space(
            start=self.v_min, stop=self.v_max, num=self.num_atoms)
        self.create_placeholders()
        self.build_graph()

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self._actor(self.states_ph)

        with tf.name_scope("actor_update"):
            probs = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            q_values = tf.reduce_sum(probs * self.z, axis=-1)
            self.policy_loss = -tf.reduce_mean(q_values)
            self.actor_update = self.get_actor_update(self.policy_loss)

        with tf.name_scope("critic_update"):
            probs = self._critic([self.states_ph, self.actions_ph])
            next_actions = self._target_actor(self.next_states_ph)
            next_probs = self._target_critic(
                [self.next_states_ph, next_actions])
            gamma = self._gamma ** self._n_step
            target_atoms = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * self.z
            tz = tf.clip_by_value(target_atoms, self.v_min, self.v_max)
            tz_z = tz[:, None, :] - self.z[None, :, None]
            tz_z = tf.clip_by_value(
                (1.0 - (tf.abs(tz_z) / self.delta_z)), 0., 1.)
            target_probs = tf.einsum("bij,bj->bi", tz_z, next_probs)
            self.value_loss = -tf.reduce_sum(
                tf.stop_gradient(target_probs) * tf.log(probs+1e-6))
            self.critic_update = self.get_critic_update(self.value_loss)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_actor_update_op = self.get_target_actor_update()
            self.target_critic_update_op = self.get_target_critic_update()
