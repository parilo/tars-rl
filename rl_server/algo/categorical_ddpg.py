# DDPG taken from
# https://github.com/nivwusquorum/tensorflow-deepq

import tensorflow as tf


class DDPG:
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
        self._gamma = tf.constant(discount_factor)
        self._target_actor_update_rate = tf.constant(target_actor_update_rate)
        self._target_critic_update_rate = tf.constant(target_critic_update_rate)

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
            self._actor_action = self._actor(self._state_for_act)

        with tf.name_scope("estimating_bellman_equation_sides"):

            # left hand side of the distributional Bellman equation
            agent_probs = self._critic([self._state, self._given_action])

            # right hand side of the Bellman equation
            self._next_action = tf.stop_gradient(self._target_actor(self._next_state))
            next_probs = tf.stop_gradient(self._target_critic([self._next_state, self._next_action]))

            discount = self._gamma ** self._n_step
            target_atoms = self._rewards[:,None] + discount * (1 - self._terminator[:,None]) * self._critic.z
            tz = tf.clip_by_value(target_atoms, self._critic.v_min, self._critic.v_max)
            tz_z = tz[:,None,:] - self._critic.z[None,:,None]
            tz_z = tf.clip_by_value((1.0 - (tf.abs(tz_z) / self._critic.delta_z)), 0., 1.)
            target_probs = tf.einsum('bij,bj->bi', tz_z, next_probs)

        with tf.name_scope("critic_update"):

            self._critic_error = -tf.reduce_sum(target_probs * tf.log(agent_probs+1e-6))

            critic_gradients = self._critic_optimizer.compute_gradients(
                self._critic_error, var_list=self._critic.variables())

            self._critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        with tf.name_scope("actor_update"):

            probs = self._critic([self._state, self._actor(self._state)])
            q_values = tf.reduce_sum(probs * self._critic.z, axis=-1)
            self._actor_error = -tf.reduce_mean(q_values)

            actor_gradients = self._actor_optimizer.compute_gradients(
                self._actor_error, var_list=self._actor.variables())
            
            actor_gradients = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var)
                               for grad, var in actor_gradients]

            self._actor_update = self._actor_optimizer.apply_gradients(actor_gradients)

        with tf.name_scope("target_networks_update"):

            self._target_actor_update = DDPG._update_target_network(
                self._actor, self._target_actor, self._target_actor_update_rate)

            self._target_critic_update = DDPG._update_target_network(
                self._critic, self._target_critic, self._target_critic_update_rate)

            self._update_all_targets = tf.group(
                self._target_actor_update, self._target_critic_update)
            
    def huber_loss(self, source, target, weights, kappa=1.0):
        err = tf.subtract(source, target)
        loss = tf.where(tf.abs(err)<kappa,
                        0.5*tf.square(err),
                        kappa*(tf.abs(err)-0.5*kappa))
        return tf.reduce_sum(tf.multiply(loss, weights))
        

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

        loss, _, _ = sess.run([self._critic_error,
                               self._critic_update,
                               self._actor_update],
                              feed_dict=feed_dict)
        return loss

    def target_network_update(self, sess):
        sess.run(self._update_all_targets)
