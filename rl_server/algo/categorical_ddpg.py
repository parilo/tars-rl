# DDPG taken from
# https://github.com/nivwusquorum/tensorflow-deepq

import tensorflow as tf
from .ddpg import BaseDDPG


class CategoricalDDPG(BaseDDPG):
    def __init__(self,
                 state_shapes,
                 action_size,
                 actor,
                 critic,
                 actor_optimizer,
                 critic_optimizer,
                 n_step=1,
                 gradient_clip=1.0,
                 discount_factor=0.99,
                 target_actor_update_rate=1.0,
                 target_critic_update_rate=1.0):

        super(CategoricalDDPG, self).__init__(state_shapes, action_size, actor, critic, actor_optimizer,
                                              critic_optimizer, n_step, gradient_clip, discount_factor,
                                              target_actor_update_rate, target_critic_update_rate)
        self._create_placeholders()
        self._create_variables()

    def _get_q_values(self, states, actions):
        probs = self._critic([states, actions])
        return tf.reduce_sum(probs * self._critic.z, axis=-1)

    def _get_critic_update(self):

        # left hand side of the distributional Bellman equation
        agent_probs = self._critic([self._state, self._given_action])

        # right hand side of the distributional Bellman equation
        next_action = self._target_actor(self._next_state)
        next_probs = self._target_critic([self._next_state, next_action])
        discount = self._gamma ** self._n_step
        target_atoms = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * self._critic.z
        tz = tf.clip_by_value(target_atoms, self._critic.v_min, self._critic.v_max)
        tz_z = tz[:, None, :] - self._critic.z[None, :, None]
        tz_z = tf.clip_by_value((1.0 - (tf.abs(tz_z) / self._critic.delta_z)), 0., 1.)
        target_probs = tf.einsum('bij,bj->bi', tz_z, next_probs)
        target_probs = tf.stop_gradient(target_probs)

        # critic gradient and update rule
        critic_loss = -tf.reduce_sum(target_probs * tf.log(agent_probs+1e-6))
        critic_gradients = self._critic_optimizer.compute_gradients(
            critic_loss, var_list=self._critic.variables())
        critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        return critic_loss, critic_update

    def _get_actor_update(self):

        # actor gradient and update rule
        probs = self._critic([self._state, self._actor(self._state)])
        q_values = tf.reduce_sum(probs * self._critic.z, axis=-1)

        actor_loss = -tf.reduce_mean(q_values)
        actor_gradients = self._actor_optimizer.compute_gradients(
            actor_loss, var_list=self._actor.variables())
        actor_gradients_clip = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var)
                                for grad, var in actor_gradients]
        actor_update = self._actor_optimizer.apply_gradients(actor_gradients_clip)
        return actor_loss, actor_update
