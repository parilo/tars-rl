import tensorflow as tf
from .base_algo import BaseAlgo


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
            placeholders=None):
        super(DDPG, self).__init__(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            n_step=n_step,
            actor_grad_val_clip=actor_grad_val_clip,
            actor_grad_norm_clip=actor_grad_norm_clip,
            critic_grad_val_clip=critic_grad_val_clip,
            critic_grad_norm_clip=critic_grad_norm_clip,
            gamma=gamma,
            target_actor_update_rate=target_actor_update_rate,
            target_critic_update_rate=target_critic_update_rate,
            scope=scope,
            placeholders=placeholders
        )

        with tf.name_scope(scope):
            self.create_placeholders()
            self.build_graph()

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self._actor(self.states_ph)
            self.gradients = self.get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            q_values = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            self.policy_loss = -tf.reduce_mean(q_values)
            self.actor_update = self.get_actor_update(self.policy_loss)

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
            self.value_loss = tf.losses.huber_loss(
                q_values, tf.stop_gradient(td_targets))
            self.critic_update = self.get_critic_update(self.value_loss)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_actor_update_op = self.get_target_actor_update()
            self.target_critic_update_op = self.get_target_critic_update()
