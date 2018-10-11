import tensorflow as tf

from .base_algo import BaseAlgo
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


def huber_loss(source, target, weights, kappa=1.0):
    err = tf.subtract(source, target)
    loss = tf.where(
        tf.abs(err) < kappa,
        0.5 * tf.square(err),
        kappa * (tf.abs(err) - 0.5 * kappa))
    return tf.reduce_mean(tf.multiply(loss, weights))


class QuantileTD3(BaseAlgo):
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
            placeholders=None):
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
        
        with tf.name_scope(scope):
            self.num_atoms = self._critic1.num_atoms
            tau_min = 1 / (2 * self.num_atoms)
            tau_max = 1 - tau_min
            self.tau = tf.lin_space(
                start=tau_min, stop=tau_max, num=self.num_atoms)
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
            q_values1 = tf.reduce_mean(self._critic1(
                [self.states_ph, self._actor(self.states_ph)]), axis=-1)
            q_values2 = tf.reduce_mean(self._critic2(
                [self.states_ph, self._actor(self.states_ph)]), axis=-1)
            q_values_min = tf.minimum(q_values1, q_values2)
            self.policy_loss = -tf.reduce_mean(q_values_min)
            self.actor_update = self.get_actor_update(self.policy_loss)

        with tf.name_scope("critic_update"):
            atoms1 = self._critic1([self.states_ph, self.actions_ph])
            atoms2 = self._critic2([self.states_ph, self.actions_ph])

            next_actions = self._target_actor(self.next_states_ph)
            actions_noise = tf.random_normal(
                tf.shape(next_actions), mean=0.0, stddev=self._act_noise_std)
            clipped_noise = tf.clip_by_value(
                actions_noise, -self._act_noise_clip, self._act_noise_clip)
            next_actions = next_actions + clipped_noise

            next_atoms1 = self._target_critic1(
                [self.next_states_ph, next_actions])
            next_atoms2 = self._target_critic2(
                [self.next_states_ph, next_actions])
            next_q_values1 = tf.reduce_mean(next_atoms1, axis=-1)
            next_q_values2 = tf.reduce_mean(next_atoms2, axis=-1)
            q_diff = next_q_values1 - next_q_values2
            mask = tf.where(
                q_diff < 0,
                tf.ones_like(q_diff),
                tf.zeros_like(q_diff))[:, None]
            next_atoms = mask * next_atoms1 + (1 - mask) * next_atoms2
            gamma = self._gamma ** self._n_step
            target_atoms = self.rewards_ph[:, None] + gamma * (
                1 - self.dones_ph[:, None]) * next_atoms
            target_atoms = tf.stop_gradient(target_atoms)

            self.value_loss = self.quantile_loss(atoms1, target_atoms)
            self.value_loss2 = self.quantile_loss(atoms2, target_atoms)

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

    def quantile_loss(self, atoms, target_atoms):
        atoms_diff = target_atoms[:, None, :] - atoms[:, :, None]
        delta_atoms_diff = tf.where(
            atoms_diff < 0,
            tf.ones_like(atoms_diff),
            tf.zeros_like(atoms_diff))
        weights = tf.abs(
            self.tau[None, :, None] - delta_atoms_diff) / self.num_atoms
        loss = huber_loss(
            atoms[:, :, None], target_atoms[:, None, :], weights)
        return loss

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
