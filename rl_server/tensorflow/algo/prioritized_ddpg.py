import tensorflow as tf
from .ddpg import BaseDDPG


class PrioritizedDDPG(BaseDDPG):
    def __init__(self,
                 state_shapes,
                 action_size,
                 actor,
                 critic,
                 actor_optimizer,
                 critic_optimizer,
                 n_step=1,
                 gradient_clip=1.0,
                 discount_factor=0.95,
                 target_actor_update_rate=1.0,
                 target_critic_update_rate=1.0):

        super(PrioritizedDDPG, self).__init__(state_shapes, action_size, actor, critic, actor_optimizer,
                                              critic_optimizer, n_step, gradient_clip, discount_factor,
                                              target_actor_update_rate, target_critic_update_rate)
        self._create_placeholders()
        self._create_variables()

    def _create_placeholders(self):
        super(PrioritizedDDPG, self)._create_placeholders()

        # additional placeholder for importance sampling weights
        self._is_weights = tf.placeholder(tf.float32, (None, ), name='importance_sampling_weights')
        
    def _get_critic_update(self):

        # left hand side of the Bellman equation
        agent_q = self._critic([self._state, self._given_action])

        # right hand side of the Bellman equation
        next_action = self._target_actor(self._next_state)
        next_q = self._target_critic([self._next_state, next_action])
        discount = self._gamma ** self._n_step
        target_q = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_q

        # critic gradient and update rule
        self._td_errors = tf.stop_gradient(agent_q - target_q)
        self._td_errors = tf.clip_by_value(self._td_errors, -self._grad_clip, self._grad_clip)
        critic_loss = tf.reduce_mean(self._is_weights * self._td_errors * agent_q)
        critic_gradients = self._critic_optimizer.compute_gradients(
            critic_loss, var_list=self._critic.variables())
        critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        return critic_loss, critic_update

    def train(self, sess, batch, is_weights):

        feed_dict = {self._rewards: batch.r,
                     self._given_action: batch.a,
                     self._terminator: batch.done,
                     self._is_weights: is_weights}

        for i in range(len(batch.s)):
            feed_dict[self._state[i]] = batch.s[i]
            feed_dict[self._next_state[i]] = batch.s_[i]

        loss, _, _ = sess.run([self._critic_loss,
                               self._critic_update,
                               self._actor_update],
                              feed_dict=feed_dict)
        return loss

    def get_td_errors(self, sess, batch):

        feed_dict = {self._rewards: batch.r,
                     self._given_action: batch.a,
                     self._terminator: batch.done}

        for i in range(len(batch.s)):
            feed_dict[self._state[i]] = batch.s[i]
            feed_dict[self._next_state[i]] = batch.s_[i]

        td_errors = sess.run(self._td_errors, feed_dict=feed_dict)
        return td_errors
