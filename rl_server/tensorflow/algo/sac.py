import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag as Normal

from .base_algo import BaseAlgo, network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


class SAC(BaseAlgo):
    def __init__(
        self,
        state_shapes,
        action_size,
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optimizer,
        critic_q1_optimizer,
        critic_q2_optimizer,
        critic_v_optimizer,
        action_squash_func=None,
        n_step=1,
        actor_grad_val_clip=None,
        actor_grad_norm_clip=None,
        critic_grad_val_clip=None,
        critic_grad_norm_clip=None,
        gamma=0.99,
        reward_scale=1.,
        mu_and_sig_reg=1e-3,
        target_critic_update_rate=1.0,
        # target_actor_update_rate=1.0,
        scope="algorithm",
        placeholders=None,
        actor_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]}
    ):
        super().__init__(
            state_shapes,
            action_size,
            placeholders,
            actor_optim_schedule,
            critic_optim_schedule,
            training_schedule
        )

        self._actor = actor
        self._critic_q1 = critic_q1
        self._critic_q2 = critic_q2
        self._critic_v = critic_v
        self._actor_weights_tool = ModelWeightsTool(actor)
        self._critic_q1_weights_tool = ModelWeightsTool(critic_q1)
        self._critic_q2_weights_tool = ModelWeightsTool(critic_q2)
        self._critic_v_weights_tool = ModelWeightsTool(critic_v)
        self._target_critic_v = critic_v.copy(scope=scope + "/target_critic")
        self._actor_optimizer = actor_optimizer
        self._critic_q1_optimizer = critic_q1_optimizer
        self._critic_q2_optimizer = critic_q2_optimizer
        self._critic_v_optimizer = critic_v_optimizer
        self._squash_func = action_squash_func or self._actor.out_activation
        self._n_step = n_step
        self._actor_grad_val_clip = actor_grad_val_clip
        self._actor_grad_norm_clip = actor_grad_norm_clip
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._mu_and_sig_reg = mu_and_sig_reg
        self._target_critic_update_rate = target_critic_update_rate

        with tf.name_scope(scope):
            self.build_graph()

    def _get_gradients_wrt_actions(self):
        q_values = self._critic_q1([self.states_ph, self.actions_ph])
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def _get_actor_update(self, loss):
        update_op = network_update(
            loss, self._actor, self._actor_optimizer,
            self._actor_grad_val_clip, self._actor_grad_norm_clip)
        return update_op

    def _get_critic_q1_update(self, loss):
        update_op = network_update(
            loss, self._critic_q1, self._critic_q1_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def _get_critic_q2_update(self, loss):
        update_op = network_update(
            loss, self._critic_q2, self._critic_q2_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def _get_critic_v_update(self, loss):
        update_op = network_update(
            loss, self._critic_v, self._critic_v_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def _get_target_critic_update(self):
        update_op = target_network_update(
            self._target_critic_v, self._critic_v,
            self._target_critic_update_rate)
        return update_op

    def _get_targets_init(self):
        critic_init = target_network_update(
            self._target_critic_v, self._critic_v, 1.0)
        return critic_init

    def _get_mu_and_log_sig(self, actor):
        actor_output = actor(self.states_ph)
        mu = actor_output[:, :self.action_size]
        log_sig = actor_output[:, self.action_size:]
        return mu, log_sig

    def _get_actions(self):
        mu, log_sig = self._get_mu_and_log_sig(self._actor)
        log_sig = tf.clip_by_value(log_sig, -5., 2.)
        _, actions = self._gauss_log_pi(mu, log_sig)
        return actions

    def _get_deterministic_actions(self):
        mu, log_sig = self._get_mu_and_log_sig(self._actor)
        actions = self._squash_actions(mu)
        return actions

    def _gauss_log_pi(self, mu, log_sig):
        sigma = tf.exp(log_sig)
        normal = Normal(mu, sigma)
        z = normal.sample()
        actions = self._squash_actions(z)
        gauss_log_prob = normal.log_prob(z)
        log_pi = gauss_log_prob - self._squash_correction(z)
        return log_pi[:, None], actions

    def _squash_actions(self, actions):
        if self._squash_func == "tanh":
            return tf.tanh(actions)
        if self._squash_func == "sigmoid":
            return tf.sigmoid(actions)
        return actions

    def _squash_correction(self, z):
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
        self.create_placeholders()
        with tf.name_scope("taking_action"):
            self._actions = self._get_actions()
            self._deterministic_actions = self._get_deterministic_actions()
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_and_v_update"):
            mu, log_sig = self._get_mu_and_log_sig(self._actor)
            log_sig = tf.clip_by_value(log_sig, -5., 2.)
            log_pi, actions = self._gauss_log_pi(mu, log_sig)
            v_values = self._critic_v(self.states_ph)
            q_values1 = self._critic_q1([self.states_ph, actions])
            q_values2 = self._critic_q2([self.states_ph, actions])
            q_values = tf.minimum(q_values1, q_values2)
            target_v_values = q_values - log_pi
            self._v_loss = 0.5 * tf.reduce_mean(
                (v_values - tf.stop_gradient(target_v_values)) ** 2)
            self._value_loss = self._v_loss
            self._critic_v_update = self._get_critic_v_update(self._v_loss)

            kl_loss = tf.reduce_mean(log_pi - q_values1)
            reg_loss = 0.5 * tf.reduce_mean(tf.square(mu))
            reg_loss += 0.5 * tf.reduce_mean(tf.square(log_sig))
            self._policy_loss = kl_loss + self._mu_and_sig_reg * reg_loss
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("q_update"):
            q_values1 = self._critic_q1([self.states_ph, self.actions_ph])
            q_values2 = self._critic_q2([self.states_ph, self.actions_ph])
            next_v_values = self._target_critic_v(self.next_states_ph)
            gamma = self._gamma ** self._n_step
            rewards = self._reward_scale * self.rewards_ph[:, None]
            td_targets = rewards + gamma * (
                1 - self.dones_ph[:, None]) * next_v_values
            self._q1_loss = 0.5 * tf.reduce_mean(
                (q_values1 - tf.stop_gradient(td_targets)) ** 2)
            self._q2_loss = 0.5 * tf.reduce_mean(
                (q_values2 - tf.stop_gradient(td_targets)) ** 2)
            self._critic_q1_update = self._get_critic_q1_update(self._q1_loss)
            self._critic_q2_update = self._get_critic_q2_update(self._q2_loss)
            self._critic_update = tf.group(
                self._critic_v_update, self._critic_q1_update, self._critic_q2_update)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_critic_update_op = self._get_target_critic_update()
            self._target_actor_update_op = tf.no_op()

    def act_batch(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self._actions, feed_dict=feed_dict)
        return actions.tolist()

    def act_batch_with_gradients(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self._actions, feed_dict=feed_dict)
        feed_dict = {**feed_dict, **{self.actions_ph: actions}}
        gradients = sess.run(self._gradients, feed_dict=feed_dict)
        return actions.tolist(), gradients.tolist()

    def act_batch_deterministic(self, sess, states, validation=False):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self._deterministic_actions, feed_dict=feed_dict)
        return actions.tolist()

    def train(self, sess, step_index, batch, actor_update=True, critic_update=True):
        actor_lr = self.get_actor_lr(step_index)
        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.actor_lr_ph: actor_lr,
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [self._q1_loss, self._v_loss, self._policy_loss]
        if critic_update:
            ops.append(self._critic_q1_update)
            ops.append(self._critic_q2_update)
            ops.append(self._critic_v_update)
        if actor_update:
            ops.append(self._actor_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        q_loss, v_loss, policy_loss = ops_[:3]
        return {
            'critic lr': critic_lr,
            'actor lr': actor_lr,
            'q loss': q_loss,
            'v loss': v_loss,
            'pi loss': policy_loss
        }

    def target_actor_update(self, sess):
        pass

    def target_critic_update(self, sess):
        sess.run(self._target_critic_update_op)

    def target_network_init(self, sess):
        sess.run(self._targets_init_op)

    def get_weights(self, sess, index=0):
        return {
            'actor': self._actor_weights_tool.get_weights(sess),
            'critic_v': self._critic_v_weights_tool.get_weights(sess),
            'critic_q1': self._critic_q1_weights_tool.get_weights(sess),
            'critic_q2': self._critic_q2_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._actor_weights_tool.set_weights(sess, weights['actor'])
        self._critic_v_weights_tool.set_weights(sess, weights['critic_v'])
        self._critic_q1_weights_tool.set_weights(sess, weights['critic_q1'])
        self._critic_q2_weights_tool.set_weights(sess, weights['critic_q2'])

    def reset_states(self):
        self._critic_v.reset_states()
        self._critic_q1.reset_states()
        self._critic_q2.reset_states()
        self._actor.reset_states()
