import tensorflow as tf

from .base_algo import BaseAlgo
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


class TD3(BaseAlgo):
    def __init__(
        self,
        state_shapes,
        action_size,
        actor,
        critic1,
        critic2,
        actor_optimizer,
        critic1_optimizer,
        critic2_optimizer,
        n_step=1,
        actor_grad_val_clip=1.0,
        actor_grad_norm_clip=None,
        critic_grad_val_clip=None,
        critic_grad_norm_clip=None,
        action_noise_std=0.02,
        action_noise_clip=0.5,
        gamma=0.99,
        target_actor_update_rate=1.0,
        target_critic_update_rate=1.0,
        scope="algorithm",
        placeholders=None,
        actor_optim_schedule=[{'limit': 0, 'lr': 1e-4}],
        critic_optim_schedule=[{'limit': 0, 'lr': 1e-4}],
        training_schedule=[{'limit': 0, 'batch_size_mult': 1}]
    ):
        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic1 = critic1
        self._critic2 = critic2
        self._actor_weights_tool = ModelWeightsTool(actor)
        self._critic1_weights_tool = ModelWeightsTool(critic1)
        self._critic2_weights_tool = ModelWeightsTool(critic2)
        self._target_actor = actor.copy(scope=scope + "/target_actor")
        self._target_critic1 = critic1.copy(scope=scope + "/target_critic1")
        self._target_critic2 = critic2.copy(scope=scope + "/target_critic2")
        self._actor_optimizer = actor_optimizer
        self._critic1_optimizer = critic1_optimizer
        self._critic2_optimizer = critic2_optimizer
        self._n_step = n_step
        self._actor_grad_val_clip = actor_grad_val_clip
        self._actor_grad_norm_clip = actor_grad_norm_clip
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._act_noise_std = action_noise_std
        self._act_noise_clip = action_noise_clip
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate
        self._placeholders = placeholders
        self._actor_optim_schedule = actor_optim_schedule
        self._critic_optim_schedule = critic_optim_schedule
        self._training_schedule = training_schedule
        
        with tf.name_scope(scope):
            self.create_placeholders()
            self.build_graph()

    def get_gradients_wrt_actions(self):
        q_values = self._critic1([self.states_ph, self.actions_ph])
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def get_critic1_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._critic1, self._critic1_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def get_critic2_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._critic2, self._critic2_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def get_target_critic1_update(self):
        update_op = BaseAlgo.target_network_update(
            self._target_critic1, self._critic1,
            self._target_critic_update_rate)
        return update_op

    def get_target_critic2_update(self):
        update_op = BaseAlgo.target_network_update(
            self._target_critic2, self._critic2,
            self._target_critic_update_rate)
        return update_op

    def get_targets_init(self):
        actor_init = BaseAlgo.target_network_update(
            self._target_actor, self._actor, 1.0)
        critic1_init = BaseAlgo.target_network_update(
            self._target_critic1, self._critic1, 1.0)
        critic2_init = BaseAlgo.target_network_update(
            self._target_critic2, self._critic2, 1.0)
        return tf.group(actor_init, critic1_init, critic2_init)

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self._actor(self.states_ph)
            self.gradients = self.get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            q_values = self._critic1(
                [self.states_ph, self._actor(self.states_ph)])
            self.policy_loss = -tf.reduce_mean(q_values)
            self.actor_update = self.get_actor_update(self.policy_loss)

        with tf.name_scope("critic_update"):
            q_values1 = self._critic1([self.states_ph, self.actions_ph])
            q_values2 = self._critic2([self.states_ph, self.actions_ph])
            next_actions = self._target_actor(self.next_states_ph)
            actions_noise = tf.random_normal(
                tf.shape(next_actions), mean=0.0, stddev=self._act_noise_std)
            clipped_noise = tf.clip_by_value(
                actions_noise, -self._act_noise_clip, self._act_noise_clip)
            next_actions = next_actions + clipped_noise
            next_q_values1 = self._target_critic1(
                [self.next_states_ph, next_actions])
            next_q_values2 = self._target_critic2(
                [self.next_states_ph, next_actions])
            next_q_values_min = tf.minimum(next_q_values1, next_q_values2)
            gamma = self._gamma ** self._n_step
            td_targets = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * next_q_values_min
            self.value_loss = tf.losses.huber_loss(
                q_values1, tf.stop_gradient(td_targets))
            self.value_loss2 = tf.losses.huber_loss(
                q_values2, tf.stop_gradient(td_targets))
            self.critic1_update = self.get_critic1_update(self.value_loss)
            self.critic2_update = self.get_critic2_update(self.value_loss2)
            self.critic_update = tf.group(
                self.critic1_update, self.critic2_update)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_actor_update_op = self.get_target_actor_update()
            self.target_critic_update_op = tf.group(
                self.get_target_critic1_update(),
                self.get_target_critic2_update())

    def _get_info(self):
        info = {}
        info["algo"] = "td3"
        info["actor"] = self._actor.get_info()
        info["critic"] = self._critic1.get_info()
        return info

    def get_weights(self, sess, index=0):
        return {
            'actor': self._actor_weights_tool.get_weights(sess),
            'critic1': self._critic1_weights_tool.get_weights(sess),
            'critic2': self._critic2_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._actor_weights_tool.set_weights(sess, weights['actor'])
        self._critic1_weights_tool.set_weights(sess, weights['critic1'])
        self._critic2_weights_tool.set_weights(sess, weights['critic2'])
