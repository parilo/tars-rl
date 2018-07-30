# DDPG taken from
# https://github.com/nivwusquorum/tensorflow-deepq

import tensorflow as tf


class BaseDDPG:
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

        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic = critic
        self._target_actor = actor.copy(scope='target_actor')
        self._target_critic = critic.copy(scope='target_critic')
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._grad_clip = gradient_clip
        self._gamma = discount_factor
        self._update_rates = [target_actor_update_rate, target_critic_update_rate]
        self._target_actor_update_rate = tf.constant(target_actor_update_rate)
        self._target_critic_update_rate = tf.constant(target_critic_update_rate)
        self._action_var = tf.Variable(tf.zeros((1, self._action_size)), dtype=tf.float32)

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

    def _get_action_for_state(self):
        return self._actor(self._state_for_act)

    def _get_q_values(self, states, actions):
        return self._critic([states, actions])

    def _get_gradients_for_action(self, action):
        self._action_assign = tf.assign(self._action_var, action)
        q_values = self._get_q_values(self._state_for_act, self._action_var)
        actor_loss = -tf.reduce_mean(q_values)
        return self._actor_optimizer.compute_gradients(actor_loss, var_list=[self._action_var])

    def _get_critic_update(self):

        # left hand side of the Bellman equation
        agent_q = self._critic([self._state, self._given_action])

        # right hand side of the Bellman equation
        next_action = self._target_actor(self._next_state)
        next_q = self._target_critic([self._next_state, next_action])
        discount = self._gamma ** self._n_step
        target_q = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_q

        # critic gradient and update rule
        critic_loss = tf.losses.huber_loss(agent_q, tf.stop_gradient(target_q))
        critic_gradients = self._critic_optimizer.compute_gradients(
            critic_loss, var_list=self._critic.variables())
        critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        return [critic_loss, tf.reduce_mean(agent_q**2)], critic_update

    def _get_actor_update(self):

        # actor gradient and update rule
        actor_loss = -tf.reduce_mean(self._critic([self._state, self._actor(self._state)]))
        actor_gradients = self._actor_optimizer.compute_gradients(
            actor_loss, var_list=self._actor.variables())
        actor_gradients_clip = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var)
                                for grad, var in actor_gradients]
        actor_update = self._actor_optimizer.apply_gradients(actor_gradients_clip)

        return actor_loss, actor_update

    def _get_target_critic_update(self):
        target_critic_update = BaseDDPG._update_target_network(
            self._critic, self._target_critic, self._target_critic_update_rate)
        return target_critic_update

    def _get_target_actor_update(self):
        target_actor_update = BaseDDPG._update_target_network(
            self._actor, self._target_actor, self._target_actor_update_rate)
        return target_actor_update

    def _get_targets_init(self):
        target_actor_update = BaseDDPG._update_target_network(self._actor, self._target_actor, 1.0)
        target_critic_update = BaseDDPG._update_target_network(self._critic, self._target_critic, 1.0)
        update_targets = tf.group(target_actor_update, target_critic_update)
        return update_targets

    def _create_variables(self):

        with tf.name_scope("taking_action"):
            self._actor_action = self._get_action_for_state()
            self._gradient_for_action = self._get_gradients_for_action(self._actor_action)

        with tf.name_scope("critic_update"):
            self._critic_loss, self._critic_update = self._get_critic_update()

        with tf.name_scope("actor_update"):
            self._actor_loss, self._actor_update = self._get_actor_update()

        with tf.name_scope("target_networks_update"):
            self._targets_init = self._get_targets_init()
            self._target_actor_update = self._get_target_actor_update()
            self._target_critic_update = self._get_target_critic_update()

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def act_batch(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions = sess.run(self._actor_action, feed_dict=feed_dict)
        return actions.tolist()

    def act_batch_with_gradients(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions, _ = sess.run([self._actor_action, self._action_assign], feed_dict=feed_dict)
        grad = sess.run(self._gradient_for_action[0][0], feed_dict=feed_dict)
        return actions.tolist(), grad.tolist()

    def train(self, sess, batch):

        feed_dict = {self._rewards: batch.r,
                     self._given_action: batch.a,
                     self._terminator: batch.done}
        for i in range(len(batch.s)):
            feed_dict[self._state[i]] = batch.s[i]
            feed_dict[self._next_state[i]] = batch.s_[i]
        loss, _, _ = sess.run([self._critic_loss,
                               self._critic_update,
                               self._actor_update],
                              feed_dict=feed_dict)
        return loss

    def target_actor_update(self, sess):
        sess.run(self._target_actor_update)
        
    def target_critic_update(self, sess):
        sess.run(self._target_critic_update)

    def target_network_init(self, sess):
        sess.run(self._targets_init)
        
    def _get_info(self):
        info = {}
        info['algo'] = 'ddpg'
        info['actor'] = self._actor.get_info()
        info['critic'] = self._critic.get_info()
        info['grad_clip'] = self._grad_clip
        info['discount_factor'] = self._gamma
        info['target_actor_update_rate'] = self._update_rates[0]
        info['target_critic_update_rate'] = self._update_rates[1]
        return info


class DDPG(BaseDDPG):
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

        super(DDPG, self).__init__(state_shapes, action_size, actor, critic, actor_optimizer,
                                   critic_optimizer, n_step, gradient_clip, discount_factor,
                                   target_actor_update_rate, target_critic_update_rate)
        self._create_placeholders()
        self._create_variables()
