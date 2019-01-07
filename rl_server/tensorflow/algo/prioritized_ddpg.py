import tensorflow as tf

from .ddpg import DDPG


class PrioritizedDDPG(DDPG):

    def build_graph(self):
        self.create_placeholders()
        self._is_weights = tf.placeholder(tf.float32, (None,), name="importance_sampling_weights")

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)
            self._gradients = self._get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            self._q_values = self._critic(
                [self.states_ph, self._actor(self.states_ph)])
            self._policy_loss = -tf.reduce_mean(self._q_values)
            self._actor_update = self._get_actor_update(self._policy_loss)

        with tf.name_scope("critic_update"):
            # left hand side of the Bellman equation
            agent_q = self._critic([self.states_ph, self.actions_ph])

            # right hand side of the Bellman equation
            next_action = self._target_actor(self.next_states_ph)
            next_q = self._target_critic([self.next_states_ph, next_action])
            discount = self._gamma ** self._n_step
            target_q = self.rewards_ph[:, None] + discount * (1 - self.dones_ph[:, None]) * next_q

            # critic gradient and update rule
            self._td_errors = tf.stop_gradient(agent_q - target_q)
            if self._critic_grad_val_clip is not None:
                self._td_errors = tf.clip_by_value(
                    self._td_errors,
                    -self._critic_grad_val_clip,
                    self._critic_grad_val_clip
                )
            self._value_loss = tf.reduce_mean(self._is_weights * self._td_errors * agent_q)
            self._critic_update = self._get_critic_update(self._value_loss)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_actor_update_op = self._get_target_actor_update()
            self._target_critic_update_op = self._get_target_critic_update()

    def train(self, sess, step_index, batch, is_weights, actor_update=True, critic_update=True):
        actor_lr = self.get_actor_lr(step_index)
        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.actor_lr_ph: actor_lr,
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done},
            self._is_weights: is_weights
        }
        ops = [self._q_values, self._value_loss, self._policy_loss]
        if critic_update:
            ops.append(self._critic_update)
        if actor_update:
            ops.append(self._actor_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        return {
            'critic lr':  critic_lr,
            'actor lr': actor_lr,
            'q values': ops_[0],
            'q loss': ops_[1],
            'pi loss': ops_[2]
        }

    def get_td_errors(self, sess, batch):

        feed_dict = {self.rewards_ph: batch.r,
                     self.actions_ph: batch.a,
                     self.dones_ph: batch.done}

        for i in range(len(batch.s)):
            feed_dict[self.states_ph[i]] = batch.s[i]
            feed_dict[self.next_states_ph[i]] = batch.s_[i]

        td_errors = sess.run(self._td_errors, feed_dict=feed_dict)
        return td_errors
