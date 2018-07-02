import tensorflow as tf


class SAC:
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
        self._target_critic_v_update_rate = tf.constant(target_critic_v_update_rate)

        self._create_placeholders()
        self._create_variables()

    @staticmethod
    def _update_target_network(source_network, target_network, update_rate):
        return tf.group(*(
            # this is equivalent to target =
            # (1-alpha) * target + alpha * source
            v_target.assign_sub(
                update_rate * (v_target - v_source))
            for v_source, v_target in zip(
                source_network.variables(),
                target_network.variables()
            )
        ))

    def _create_placeholders(self):
        state_batch_shapes = []
        for s in self._state_shapes:
            state_batch_shapes.append(tuple([None] + list(s)))

        self._rewards = tf.placeholder(tf.float32, (None, ), name='inp_rewards')
        self._given_action = tf.placeholder(tf.float32, (None, self._action_size), name='inp_actions')
        self._state_for_act = []
        self._state = []
        self._next_state = []
        for shape in state_batch_shapes:
            self._state_for_act.append(tf.placeholder(tf.float32, shape, name='inps_states_for_act'))
            self._state.append(tf.placeholder(tf.float32, shape, name='inp_prev_state'))
            self._next_state.append(tf.placeholder(tf.float32, shape, name='inp_next_states'))
        self._terminator = tf.placeholder(tf.float32, (None, ), name='inp_terminator')

    def _create_variables(self):

        with tf.name_scope("taking_action"):
            agent_log_weights, agent_mu, agent_log_std = self._actor(self._state_for_act)
            agent_log_std = tf.clip_by_value(agent_log_std, -20., 2.)
            _, self._actor_action = self.gmm_log_pi(agent_log_weights, agent_mu, agent_log_std)

        with tf.name_scope("estimating_log_pi"):
            log_weights, mu, log_std = self._actor(self._state)
            log_std = tf.clip_by_value(log_std, -20., 2.)
            log_pi, sampled_action = self.gmm_log_pi(log_weights, mu, log_std)

        with tf.name_scope("critic_v_update"):
            v = self._critic_v(self._state)
            q_on_policy = self._critic_q([self._state, sampled_action])
            v_target = tf.stop_gradient(q_on_policy - log_pi)
            self._critic_v_error = 0.5*tf.reduce_mean((v - v_target)**2)

            critic_v_gradients = self._critic_v_optimizer.compute_gradients(
                self._critic_v_error, var_list=self._critic_v.variables())
            self._critic_v_update = self._critic_v_optimizer.apply_gradients(critic_v_gradients)

        with tf.name_scope("critic_q_update"):
            q_off_policy = self._critic_q([self._state, self._given_action])
            next_v = self._target_critic_v(self._next_state)
            discount = self._gamma ** self._n_step
            q_target = tf.stop_gradient(self._rewards[:,None]/self._temp + discount * (1 - self._terminator[:,None]) * next_v)
            self._critic_q_error = 0.5*tf.reduce_mean((q_off_policy - q_target)**2)

            critic_q_gradients = self._critic_q_optimizer.compute_gradients(
                self._critic_q_error, var_list=self._critic_q.variables())
            self._critic_q_update = self._critic_q_optimizer.apply_gradients(critic_q_gradients)

        with tf.name_scope("actor_update"):
            log_pi_target = tf.stop_gradient(q_on_policy - v)
            self._actor_error = 0.5*tf.reduce_mean((log_pi - log_pi_target)**2)
            #self._actor_error += 1e-3*tf.reduce_mean(mu**2) + 1e-3*tf.reduce_mean(log_sigma**2)

            actor_gradients = self._actor_optimizer.compute_gradients(
                self._actor_error, var_list=self._actor.variables())
            self._actor_update = self._actor_optimizer.apply_gradients(actor_gradients)

        with tf.name_scope("target_networks_update"):
            self._update_all_targets = SAC._update_target_network(
                self._critic_v, self._target_critic_v, self._target_critic_v_update_rate)    
            
    def gmm_log_pi(self, log_weights, mu, log_std):
        sigma = tf.exp(log_std)
        normal = tf.distributions.Normal(mu, sigma)
        
        z = normal.sample()
        sample_w = tf.reshape(tf.multinomial(logits=log_weights, num_samples=1), (-1,))
        w_onehot = tf.one_hot(sample_w, depth=self._actor.K)
        chosen_z = tf.stop_gradient(tf.reduce_sum(z * w_onehot[:,:,None], axis=1))
        actions = self.squash_action(chosen_z)
        
        log_z = normal.log_prob(chosen_z[:,None,:])
        powers = log_z + log_weights[:,:,None]
        
        log_pi = tf.reduce_logsumexp(powers, axis=-1)
        log_pi -= tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)
        log_pi -= self.get_squash_correction(chosen_z)
        return log_pi, actions
    
    def squash_action(self, action):
        if self._actor.out_activation == 'tanh':
            return tf.tanh(action)
        if self._actor.out_activation == 'sigmoid':
            return tf.sigmoid(action)
        return action
    
    def get_squash_correction(self, action):
        if self._actor.out_activation == 'tanh':
            corr = tf.log(1 - tf.tanh(action)**2 + 1e-6)
            return tf.reduce_sum(corr, axis=-1, keepdims=True)
        if self._actor.out_activation == 'sigmoid':
            corr = tf.log(tf.sigmoid(action)*tf.sigmoid(-action) + 1e-6)
            return tf.reduce_sum(corr, axis=-1, keepdims=True)
        return 0

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def act_batch(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions = sess.run(self._actor_action, feed_dict=feed_dict)
        return actions.tolist()

    def train(self, sess, batch):

        feed_dict = {self._rewards: batch.r,
                     self._given_action: batch.a,
                     self._terminator: batch.done}

        for i in range(len(batch.s)):
            feed_dict[self._state[i]] = batch.s[i]
            feed_dict[self._next_state[i]] = batch.s_[i]
            
        loss, v, q, p, = sess.run([self._critic_v_error,
                                   self._critic_v_update,
                                   self._critic_q_update,
                                   self._actor_update],
                                  feed_dict=feed_dict)
        return loss

    def target_network_update(self, sess):
        sess.run(self._update_all_targets)
