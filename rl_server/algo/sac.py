import tensorflow as tf
from .ddpg import BaseDDPG


class SAC(BaseDDPG):
    def __init__(self,
                 state_shapes,
                 action_size,
                 actor,
                 critic_v,
                 critic_q,
                 actor_optimizer,
                 critic_v_optimizer,
                 critic_q_optimizer,
                 n_step=1,
                 gradient_clip=1.0,
                 discount_factor=0.99,
                 temperature=1e-2,
                 mean_and_std_reg=0.0,
                 target_critic_v_update_rate=1.0):

        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic_v = critic_v
        self._critic_q = critic_q
        self._target_critic_v = self._critic_v.copy(scope='target_critic_v')
        self._actor_optimizer = actor_optimizer
        self._critic_v_optimizer = critic_v_optimizer
        self._critic_q_optimizer = critic_q_optimizer
        self._n_step = n_step
        self._grad_clip = gradient_clip
        self._gamma = tf.constant(discount_factor)
        self._temp = temperature
        self._mean_and_std_reg = mean_and_std_reg
        self._target_critic_v_update_rate = tf.constant(target_critic_v_update_rate)

        self._create_placeholders()
        self._create_variables()

    def _get_action_for_state(self):
        agent_log_weights, agent_mu, agent_log_std = self._actor(self._state_for_act)
        agent_log_weights = tf.clip_by_value(agent_log_weights, -10., 2.)
        agent_log_std = tf.clip_by_value(agent_log_std, -5., 2.)
        _, action = self.gmm_log_pi(agent_log_weights, agent_mu, agent_log_std)
        print (self._state_for_act, agent_mu, action)
        return action
    
    def _get_det_action_for_state(self):
        agent_log_weights, agent_mu, agent_log_std = self._actor(self._state_for_act)
        states_batch = []
        for i, s in enumerate(self._state_shapes):
            tile_shape = [self._actor.K] + [1] * len(s)
            tiled_state = tf.tile(self._state_for_act[i], tile_shape)
            states_batch.append(tiled_state)
        actions_batch = tf.reshape(agent_mu, [-1, self._action_size])
        q_values = self._critic_q([states_batch, actions_batch])
        action = actions_batch[tf.argmax(q_values, axis=0)[0]][None, :]
        return action

    def _get_critic_update(self):

        # left hand side of the Bellman equation
        agent_q = self._critic_q([self._state, self._given_action])

        # right hand side of the Bellman equation
        next_v = self._target_critic_v(self._next_state)
        discount = self._gamma ** self._n_step
        target_q = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_v

        # critic q gradient and update rule
        critic_q_loss = 0.5 * tf.reduce_mean(tf.square(agent_q - tf.stop_gradient(target_q)))
        critic_q_gradients = self._critic_q_optimizer.compute_gradients(
            critic_q_loss, var_list=self._critic_q.variables())
        critic_q_update = self._critic_q_optimizer.apply_gradients(critic_q_gradients)

        return [critic_q_loss, tf.reduce_mean(agent_q**2)], critic_q_update

    def _get_actor_update(self):

        # estimating log pi
        log_weights, mu, log_std = self._actor(self._state)
        log_weights = tf.clip_by_value(log_weights, -10., 2.)
        log_std = tf.clip_by_value(log_std, -5., 2.)
        log_pi, sampled_action = self.gmm_log_pi(log_weights, mu, log_std)

        # critic v gradient and update rule
        agent_v = self._critic_v(self._state)
        target_q = self._critic_q([self._state, sampled_action])
        target_v = target_q - log_pi
        critic_v_loss = 0.5 * tf.reduce_mean(tf.square(agent_v - tf.stop_gradient(target_v)))
        critic_v_gradients = self._critic_v_optimizer.compute_gradients(
            critic_v_loss, var_list=self._critic_v.variables())
        critic_v_update = self._critic_v_optimizer.apply_gradients(critic_v_gradients)

        # actor gradient and update rule
        target_log_pi = target_q - agent_v
        actor_loss = 0.5 * tf.reduce_mean(tf.square(log_pi - tf.stop_gradient(target_log_pi)))
        actor_loss += self._mean_and_std_reg * (tf.reduce_mean(mu**2) + tf.reduce_mean(log_std**2))
        actor_gradients = self._actor_optimizer.compute_gradients(
            actor_loss, var_list=self._actor.variables())
        actor_update = self._actor_optimizer.apply_gradients(actor_gradients)

        return actor_loss, tf.group(critic_v_update, actor_update)

    def _get_targets_update(self):
        update_targets = BaseDDPG._update_target_network(
            self._critic_v, self._target_critic_v, self._target_critic_v_update_rate)
        return update_targets
    
    def _create_variables(self):

        with tf.name_scope("taking_action"):
            self._actor_action = self._get_action_for_state()

        with tf.name_scope("taking_deterministic_action"):
            self._actor_action_det = self._get_det_action_for_state()

        with tf.name_scope("critic_update"):
            self._critic_loss, self._critic_update = self._get_critic_update()

        with tf.name_scope("actor_update"):
            self._actor_loss, self._actor_update = self._get_actor_update()

        with tf.name_scope("target_networks_update"):
            self._targets_update = self._get_targets_update()

    def gmm_log_pi(self, log_weights, mu, log_std):
        sigma = tf.exp(log_std)
        normal = tf.distributions.Normal(mu, sigma)

        z = normal.sample()
        sample_w = tf.reshape(tf.multinomial(logits=log_weights, num_samples=1), (-1,))
        onehot_w = tf.one_hot(sample_w, depth=self._actor.K)
        chosen_z = tf.stop_gradient(tf.reduce_sum(z * onehot_w[:, :, None], axis=1))
        action = self.squash_action(chosen_z)

        log_z = normal.log_prob(chosen_z[:, None, :])
        powers = log_z + log_weights[:, :, None]

        log_pi = tf.reduce_logsumexp(powers, axis=-1)
        log_pi -= tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)
        log_pi -= self.get_squash_correction(action)
        log_pi *= self._temp
        return log_pi, action

    def squash_action(self, action):
        if self._actor.out_activation == 'tanh':
            return tf.tanh(action)
        if self._actor.out_activation == 'sigmoid':
            return tf.sigmoid(action)
        return action

    def get_squash_correction(self, action):
        if self._actor.out_activation == 'tanh':
            corr = tf.log(1 - action**2 + 1e-6)
            return tf.reduce_sum(corr, axis=-1, keepdims=True)
        if self._actor.out_activation == 'sigmoid':
            corr = tf.log(action * (1 - action) + 1e-6)
            return tf.reduce_sum(corr, axis=-1, keepdims=True)
        return 0
    
    def act_batch_deterministic(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions = sess.run(self._actor_action_det, feed_dict=feed_dict)
        return actions.tolist()
