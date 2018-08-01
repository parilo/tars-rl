import tensorflow as tf
from .ddpg import BaseDDPG


class QuantileDDPG(BaseDDPG):
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

        super(QuantileDDPG, self).__init__(state_shapes, action_size, actor, critic, actor_optimizer,
                                           critic_optimizer, n_step, gradient_clip, discount_factor,
                                           target_actor_update_rate, target_critic_update_rate)
        self._create_placeholders()
        self._create_variables()

    def _get_q_values(self, states, actions):
        atoms = self._critic([states, actions])
        return tf.reduce_mean(atoms, axis=-1)

    def _get_critic_update(self):

        # left hand side of the distributional Bellman equation
        agent_atoms = self._critic([self._state, self._given_action])

        # right hand side of the distributional Bellman equation
        next_action = self._target_actor(self._next_state)
        next_atoms = self._target_critic([self._next_state, next_action])
        discount = self._gamma ** self._n_step
        target_atoms = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_atoms
        target_atoms = tf.stop_gradient(target_atoms)

        # critic gradient and update rule
        atoms_diff = target_atoms[:, None, :] - agent_atoms[:, :, None]
        delta_atoms_diff = tf.where(atoms_diff < 0, tf.ones_like(atoms_diff), tf.zeros_like(atoms_diff))
        huber_weights = tf.abs(self._critic.tau[None, :, None] - delta_atoms_diff) / self._critic.num_atoms
        critic_loss = self.huber_loss(agent_atoms[:, :, None], target_atoms[:, None, :], huber_weights)
        critic_gradients = self._critic_optimizer.compute_gradients(
            critic_loss, var_list=self._critic.variables())
        critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        average_q = tf.reduce_mean((tf.reduce_sum(agent_atoms, axis=-1) / self._critic.num_atoms)**2)
        return [critic_loss, average_q], critic_update

    def huber_loss(self, source, target, weights, kappa=1.0):
        err = tf.subtract(source, target)
        loss = tf.where(tf.abs(err) < kappa,
                        0.5 * tf.square(err),
                        kappa * (tf.abs(err) - 0.5 * kappa))
        return tf.reduce_mean(tf.multiply(loss, weights))
