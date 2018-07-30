import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag as Normal
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
                 mu_and_sig_reg=0.0,
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
        self._gamma = discount_factor
        self._temp = temperature
        self._mu_and_sig_reg = mu_and_sig_reg
        self._update_rates = [target_critic_v_update_rate]
        self._target_critic_v_update_rate = tf.constant(target_critic_v_update_rate)
        self._action_var = tf.Variable(tf.zeros((1, self._action_size)), dtype=tf.float32)

        self._create_placeholders()
        self._create_variables()

    def _get_action_for_state(self):
        agent_log_weights, agent_mu, agent_log_std = self._actor(self._state_for_act)
        agent_log_weights = tf.maximum(agent_log_weights, -10.)
        agent_log_std = tf.clip_by_value(agent_log_std, -5., 2.)
        _, action = self.gmm_log_pi(agent_log_weights, agent_mu, agent_log_std)
        return action

    def _get_q_values(self, states, actions):
        return self._critic_q([states, actions])

    def _get_det_action_for_state(self):
        """sample best action from gaussian means"""
        agent_log_weights, agent_mu, agent_log_std = self._actor(self._state_for_act)
        agent_mu = self.squash_action(agent_mu)
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
        agent_log_weights, agent_mu, agent_log_sig = self._actor(self._state)
        agent_log_sig = tf.clip_by_value(agent_log_sig, -5., 2.)
        agent_log_weights = tf.maximum(agent_log_weights, -10.)
        agent_log_pi, agent_action = self.gmm_log_pi(agent_log_weights, agent_mu, agent_log_sig)

        # critic v gradient and update rule
        agent_v = self._critic_v(self._state)
        target_q = self._critic_q([self._state, agent_action])
        target_v = target_q - agent_log_pi
        critic_v_loss = 0.5 * tf.reduce_mean(tf.square(agent_v - tf.stop_gradient(target_v)))
        critic_v_gradients = self._critic_v_optimizer.compute_gradients(
            critic_v_loss, var_list=self._critic_v.variables())
        critic_v_update = self._critic_v_optimizer.apply_gradients(critic_v_gradients)

        # actor gradient and update rule
        target_log_pi = target_q - agent_v
        policy_loss = tf.reduce_mean(agent_log_pi * tf.stop_gradient(agent_log_pi - target_log_pi)) / self._temp
        reg_loss = 0
        reg_loss += self._mu_and_sig_reg * 0.5 * tf.reduce_mean(agent_mu**2)
        reg_loss += self._mu_and_sig_reg * 0.5 * tf.reduce_mean(agent_log_sig**2)
        actor_loss = policy_loss + reg_loss

        actor_gradients = self._actor_optimizer.compute_gradients(
            actor_loss, var_list=self._actor.variables())
        actor_gradients_clip = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var)
                                for grad, var in actor_gradients]
        actor_update = self._actor_optimizer.apply_gradients(actor_gradients_clip)

        return [tf.reduce_mean(agent_log_pi), 
                tf.reduce_mean(agent_log_sig),
                tf.reduce_mean(agent_mu)], tf.group(critic_v_update, actor_update)

    def _get_target_critic_update(self):
        target_critic_update = BaseDDPG._update_target_network(
            self._critic_v, self._target_critic_v, self._target_critic_v_update_rate)
        return target_critic_update

    def _get_targets_init(self):
        update_targets = BaseDDPG._update_target_network(
            self._critic_v, self._target_critic_v, 1.0)
        return update_targets

    def _create_variables(self):

        with tf.name_scope("taking_action"):
            self._actor_action = self._get_action_for_state()
            self._actor_action_det = self._get_det_action_for_state()
            self._gradient_for_action = self._get_gradients_for_action(self._actor_action)

        with tf.name_scope("critic_update"):
            self._critic_loss, self._critic_update = self._get_critic_update()

        with tf.name_scope("actor_update"):
            self._actor_loss, self._actor_update = self._get_actor_update()
            self._critic_loss += self._actor_loss

        with tf.name_scope("target_networks_update"):
            self._targets_init = self._get_targets_init()
            self._target_critic_update = self._get_target_critic_update()

    def gmm_log_pi(self, log_weights, mu, log_std):

        sigma = tf.exp(log_std)
        normal = Normal(mu, sigma)

        # sample from GMM
        sample_w = tf.stop_gradient(tf.multinomial(logits=log_weights, num_samples=1))
        sample_z = tf.stop_gradient(normal.sample())
        mask = tf.one_hot(sample_w[:, 0], depth=self._actor.K)
        z = tf.reduce_sum(sample_z * mask[:, :, None], axis=1)
        action = self.squash_action(z)

        # calculate log policy
        gauss_log_pi = normal.log_prob(z[:, None, :])
        log_pi = tf.reduce_logsumexp(gauss_log_pi + log_weights, axis=-1)
        log_pi -= tf.reduce_logsumexp(log_weights, axis=-1)
        log_pi -= self.get_squash_correction(z)
        log_pi *= self._temp

        return log_pi[:, None], action

    def squash_action(self, action):
        if self._actor.out_activation == 'tanh':
            return tf.tanh(action)
        if self._actor.out_activation == 'sigmoid':
            return tf.sigmoid(action)
        return action

    def get_squash_correction(self, z):
        if self._actor.out_activation == 'tanh':
            zz = tf.stack((z, -z), axis=2)
            corr = tf.log(4.) - 2*tf.reduce_logsumexp(zz, axis=-1)
            corr = tf.reduce_sum(corr, axis=-1)
            return corr
        if self._actor.out_activation == 'sigmoid':
            zz = tf.stack((tf.zeros_like(z), -z), axis=2)
            corr = -z - 2*tf.reduce_logsumexp(zz, axis=-1)
            corr = tf.reduce_sum(corr, axis=-1)
            return corr
        return 0

    def act_batch_deterministic(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions = sess.run(self._actor_action_det, feed_dict=feed_dict)
        return actions.tolist()

    def target_actor_update(self, sess):
        pass

    def _get_info(self):
        info = {}
        info['algo'] = 'sac'
        info['actor'] = self._actor.get_info()
        info['critic_v'] = self._critic_v.get_info()
        info['critic_q'] = self._critic_q.get_info()
        info['grad_clip'] = self._grad_clip
        info['discount_factor'] = self._gamma
        info['target_critic_v_update_rate'] = self._update_rates[0]
        info['temperature'] = self._temp
        info['regularization_coef'] = self._mu_and_sig_reg
        return info
