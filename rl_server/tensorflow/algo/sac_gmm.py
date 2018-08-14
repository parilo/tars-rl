import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag as Normal
from .base_algo import BaseAlgo


class SAC(BaseAlgo):
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic_q,
            critic_v,
            actor_optimizer,
            critic_q_optimizer,
            critic_v_optimizer,
            action_squash_func=None,
            n_step=1,
            actor_grad_val_clip=1.0,
            actor_grad_norm_clip=None,
            critic_grad_val_clip=None,
            critic_grad_norm_clip=None,
            gamma=0.99,
            reward_scale=1.,
            mu_and_sig_reg=1e-3,
            target_critic_update_rate=1.0):
        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic_q = critic_q
        self._critic_v = critic_v
        self._target_critic_v = critic_v.copy(scope="target_critic")
        self._actor_optimizer = actor_optimizer
        self._critic_q_optimizer = critic_q_optimizer
        self._critic_v_optimizer = critic_v_optimizer
        self._squash_func = action_squash_func or self._actor.out_activation
        self._n_step = n_step
        self._actor_grad_val_clip = actor_grad_val_clip
        self._actor_grad_norm_clip = actor_grad_norm_clip
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._num_components = self._actor.K
        self._reward_scale = reward_scale
        self._mu_and_sig_reg = mu_and_sig_reg
        self._target_critic_update_rate = target_critic_update_rate
        self.create_placeholders()
        self.build_graph()

    def get_gradients_wrt_actions(self):
        q_values = self._critic_q([self.states_ph, self.actions_ph])
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def get_actor_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._actor, self._actor_optimizer,
            self._actor_grad_val_clip, self._actor_grad_norm_clip)
        return update_op

    def get_critic_q_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._critic_q, self._critic_q_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def get_critic_v_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._critic_v, self._critic_v_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def get_target_critic_update(self):
        update_op = BaseAlgo.target_network_update(
            self._target_critic_v, self._critic_v,
            self._target_critic_update_rate)
        return update_op

    def get_targets_init(self):
        critic_init = BaseAlgo.target_network_update(
            self._target_critic_v, self._critic_v, 1.0)
        return critic_init

    def get_actions(self):
        log_w, mu, log_sig = self._actor(self.states_ph)
        log_w = tf.maximum(log_w, -10.)
        log_sig = tf.clip_by_value(log_sig, -5., 2.)
        _, actions = self.gmm_log_pi(log_w, mu, log_sig)
        return actions

    def get_deterministic_actions(self):
        log_w, mu, log_sig = self._actor(self.states_ph)
        mu = self.squash_actions(mu)
        batch_of_states = []
        for i, s in enumerate(self._state_shapes):
            tile_shape = [self._num_components] + [1] * len(s)
            tiled_state = tf.tile(self.states_ph[i], tile_shape)
            batch_of_states.append(tiled_state)
        batch_of_actions = tf.reshape(mu, [-1, self._action_size])
        q_values = self._critic_q([batch_of_states, batch_of_actions])
        actions = batch_of_actions[tf.argmax(q_values, axis=0)[0]][None, :]
        return actions

    def gmm_log_pi(self, log_w, mu, log_sig):
        sigma = tf.exp(log_sig)
        normal = Normal(mu, sigma)
        sample_log_w = tf.stop_gradient(
            tf.multinomial(logits=log_w, num_samples=1))
        sample_z = tf.stop_gradient(normal.sample())
        mask = tf.one_hot(sample_log_w[:, 0], depth=self._num_components)
        z = tf.reduce_sum(sample_z * mask[:, :, None], axis=1)
        actions = self.squash_actions(z)
        gauss_log_pi = normal.log_prob(z[:, None, :])
        log_pi = tf.reduce_logsumexp(gauss_log_pi + log_w, axis=-1)
        log_pi -= tf.reduce_logsumexp(log_w, axis=-1)
        log_pi -= self.squash_correction(z)
        return log_pi[:, None], actions

    def squash_actions(self, actions):
        if self._squash_func == "tanh":
            return tf.tanh(actions)
        if self._squash_func == "sigmoid":
            return tf.sigmoid(actions)
        return actions

    def squash_correction(self, z):
        corr = 0
        if self._squash_func == "tanh":
            zz = tf.stack((z, -z), axis=-1)
            corr = tf.log(4.) - 2 * tf.reduce_logsumexp(zz, axis=2)
            corr = tf.reduce_sum(corr, axis=-1)
        if self._squash_func == "sigmoid":
            zz = tf.stack((tf.zeros_like(z), -z), axis=2)
            corr = -z - 2 * tf.reduce_logsumexp(zz, axis=-1)
            corr = tf.reduce_sum(corr, axis=-1)
        return corr

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self.get_actions()
            self.deterministic_actions = self.get_deterministic_actions()
            self.gradients = self.get_gradients_wrt_actions()

        with tf.name_scope("actor_and_v_update"):
            log_w, mu, log_sig = self._actor(self.states_ph)
            log_w = tf.maximum(log_w, -10.)
            log_sig = tf.clip_by_value(log_sig, -5., 2.)
            log_pi, actions = self.gmm_log_pi(log_w, mu, log_sig)
            v_values = self._critic_v(self.states_ph)
            q_values = self._critic_q([self.states_ph, actions])
            target_v_values = q_values - log_pi
            self.v_value_loss = 0.5 * tf.reduce_mean(
                (v_values - tf.stop_gradient(target_v_values)) ** 2)
            self.critic_v_update = self.get_critic_v_update(self.v_value_loss)

            target_log_pi = q_values - v_values
            kl_loss = log_pi * tf.stop_gradient(log_pi - target_log_pi)
            reg_loss = 0.5 * tf.reduce_mean(tf.square(mu))
            reg_loss += 0.5 * tf.reduce_mean(tf.square(log_sig))
            self.policy_loss = kl_loss + self._mu_and_sig_reg * reg_loss
            self.actor_update = self.get_actor_update(self.policy_loss)

        with tf.name_scope("critic_update"):
            q_values = self._critic_q([self.states_ph, self.actions_ph])
            next_v_values = self._target_critic_v(self.next_states_ph)
            gamma = self._gamma ** self._n_step
            rewards = self._reward_scale * self.rewards_ph[:, None]
            td_targets = rewards + gamma * (
                1 - self.dones_ph[:, None]) * next_v_values
            self.q_value_loss = 0.5 * tf.reduce_mean(
                (q_values - tf.stop_gradient(td_targets)) ** 2)
            self.critic_q_update = self.get_critic_q_update(self.q_value_loss)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_critic_update_op = self.get_target_critic_update()

    def act_batch_deterministic(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self.deterministic_actions, feed_dict=feed_dict)
        return actions.tolist()

    def train(self, sess, batch, actor_update=True, critic_update=True):
        feed_dict = {
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [self.q_value_loss, self.v_value_loss, self.policy_loss]
        if critic_update:
            ops.append(self.critic_q_update)
            ops.append(self.critic_v_update)
        if actor_update:
            ops.append(self.actor_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        q_value_loss, v_value_loss, policy_loss = ops_[:3]
        return q_value_loss

    def target_actor_update(self, sess):
        pass

    def target_critic_update(self, sess):
        sess.run(self.target_critic_update_op)

    def target_network_init(self, sess):
        sess.run(self.targets_init_op)

    def _get_info(self):
        info = {}
        info["algo"] = "sac"
        info["actor"] = self._actor.get_info()
        return info
