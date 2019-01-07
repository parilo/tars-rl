import tensorflow as tf

from .base_algo import BaseAlgo, network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


class DDPG(BaseAlgo):
    def __init__(
        self,
        state_shapes,
        action_size,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        n_step=1,
        actor_grad_val_clip=1.0,
        actor_grad_norm_clip=None,
        critic_grad_val_clip=None,
        critic_grad_norm_clip=None,
        gamma=0.99,
        target_actor_update_rate=1.0,
        target_critic_update_rate=1.0,
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
        self._critic = critic
        self._target_actor = actor.copy(scope=scope + "/target_actor")
        self._target_critic = critic.copy(scope=scope + "/target_critic")
        self._actor_weights_tool = ModelWeightsTool(actor)
        self._critic_weights_tool = ModelWeightsTool(critic)
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._actor_grad_val_clip = actor_grad_val_clip
        self._actor_grad_norm_clip = actor_grad_norm_clip
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate

        with tf.name_scope(scope):
            self.build_graph()

    def _get_gradients_wrt_actions(self):
        q_values = self._critic([self.states_ph, self.actions_ph])
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def _get_actor_update(self, loss):
        update_op = network_update(
            loss, self._actor, self._actor_optimizer,
            self._actor_grad_val_clip, self._actor_grad_norm_clip)
        return update_op

    def _get_critic_update(self, loss):
        update_op = network_update(
            loss, self._critic, self._critic_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def _get_target_actor_update(self):
        update_op = target_network_update(
            self._target_actor, self._actor,
            self._target_actor_update_rate)
        return update_op

    def _get_target_critic_update(self):
        update_op = target_network_update(
            self._target_critic, self._critic,
            self._target_critic_update_rate)
        return update_op

    def _get_targets_init(self):
        actor_init = target_network_update(
            self._target_actor, self._actor, 1.0)
        critic_init = target_network_update(
            self._target_critic, self._critic, 1.0)
        return tf.group(actor_init, critic_init)

    def build_graph(self):
        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            self._q_values = self._critic([self.states_ph, self._actor(self.states_ph)])
            self._policy_loss = -tf.reduce_mean(self._q_values)
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("critic_update"):
            q_values = self._critic([self.states_ph, self.actions_ph])
            next_actions = self._actor(self.next_states_ph)
            # next_actions = self._target_actor(self.next_states_ph)
            next_q_values = self._target_critic(
                [self.next_states_ph, next_actions])
            # print(self._gamma, self._n_step)
            gamma = self._gamma ** self._n_step
            td_targets = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * next_q_values
            self._value_loss = tf.losses.huber_loss(
                q_values, tf.stop_gradient(td_targets))
            self._critic_update = self._get_critic_update(self._value_loss)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_actor_update_op = self._get_target_actor_update()
            self._target_critic_update_op = self._get_target_critic_update()

    def target_network_init(self, sess):
        sess.run(self._targets_init_op)

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
        ops = [self._value_loss, self._policy_loss]
        if critic_update:
            ops.append(self._critic_update)
        if actor_update:
            ops.append(self._actor_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        return {
            'critic lr':  critic_lr,
            'actor lr': actor_lr,
            'q loss': ops_[0],
            'pi loss': ops_[1]
        }

    def target_actor_update(self, sess):
        sess.run(self._target_actor_update_op)

    def target_critic_update(self, sess):
        sess.run(self._target_critic_update_op)

    def get_weights(self, sess, index=0):
        return {
            'actor': self._actor_weights_tool.get_weights(sess),
            'critic': self._critic_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._actor_weights_tool.set_weights(sess, weights['actor'])
        self._critic_weights_tool.set_weights(sess, weights['critic'])
