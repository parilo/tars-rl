import tensorflow as tf

from .base_algo_discrete import BaseAlgoDiscrete
from .base_algo import network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool
from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.tensorflow.networks.network_keras import NetworkKeras
from rl_server.tensorflow.algo.base_algo_discrete import create_placeholders


def create_algo(algo_config, placeholders, scope_postfix):

    _, _, state_shapes, action_size = algo_config.get_env_shapes()
    if placeholders is None:
        placeholders = create_placeholders(state_shapes)
    algo_scope = 'dqn_' + scope_postfix
    critic_lr = placeholders[0]

    critic_params = get_network_params(algo_config, "critic")
    critic_optim_info = algo_config.as_obj()['critic_optim']

    critic_1 = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **critic_params,
        scope='critic_1'
    )

    critic_2 = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **critic_params,
        scope='critic_2'
    )

    return TD_DQN(
        state_shapes=state_shapes,
        action_size=action_size,
        critic_1=critic_1,
        critic_2=critic_2,
        critic_optimizer=get_optimizer_class(critic_optim_info)(
            learning_rate=critic_lr),
        **algo_config.as_obj()["algorithm"],
        scope=algo_scope,
        placeholders=placeholders,
        critic_optim_schedule=critic_optim_info,
        training_schedule=algo_config.as_obj()["training"])


class TD_DQN(BaseAlgoDiscrete):
    """
    Twin Delayed DQN, done by analogue to
    TD3 https://arxiv.org/abs/1802.09477
    """
    def __init__(
        self,
        state_shapes,
        action_size,
        critic_1,
        critic_2,
        critic_optimizer,
        n_step=1,
        critic_grad_val_clip=None,
        critic_grad_norm_clip=None,
        gamma=0.99,
        target_critic_update_rate=1.0,
        scope="algorithm",
        placeholders=None,
        critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]}
    ):
        super().__init__(
            state_shapes,
            action_size,
            placeholders,
            critic_optim_schedule,
            training_schedule
        )

        self._critic_1 = critic_1
        self._critic_2 = critic_2
        self._target_critic_1 = critic_1.copy(scope=scope + "/target_critic_1")
        self._target_critic_2 = critic_2.copy(scope=scope + "/target_critic_2")
        self._critic_weights_tool_1 = ModelWeightsTool(critic_1)
        self._critic_weights_tool_2 = ModelWeightsTool(critic_2)
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._target_critic_update_rate = target_critic_update_rate

        with tf.name_scope(scope):
            self.build_graph()

    def _get_critic_update(self, critic, loss):
        update_op = network_update(
            loss,
            critic,
            self._critic_optimizer,
            self._critic_grad_val_clip,
            self._critic_grad_norm_clip
        )
        return update_op

    def _get_target_critic_update(self, target_critic, critic):
        update_op = target_network_update(
            target_critic, critic,
            self._target_critic_update_rate)
        return update_op

    def _get_targets_init(self):
        critic_1_init = target_network_update(
            self._target_critic_1,
            self._critic_1,
            1.0
        )
        critic_2_init = target_network_update(
            self._target_critic_2,
            self._critic_2,
            1.0
        )
        return tf.group([critic_1_init, critic_2_init])

    def build_graph(self):
        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._q_values_1 = self._critic_1(self.states_ph)
            self._q_values_2 = self._critic_2(self.states_ph)
            self._q_values = tf.minimum(self._q_values_1, self._q_values_2)

        with tf.name_scope("critic_update"):
            next_q_values_1 = self._target_critic_1(self.next_states_ph)
            next_q_values_2 = self._target_critic_2(self.next_states_ph)
            gamma = self._gamma ** self._n_step
            td_targets = self.rewards_ph + gamma * (1 - self.dones_ph) * tf.reduce_max(
                tf.minimum(next_q_values_1, next_q_values_2), axis=1
            )

            indices_range = tf.range(tf.shape(self.actions_ph)[0])
            action_indices = tf.stack([indices_range, self.actions_ph], axis=1)
            q_values_selected_1 = tf.gather_nd(self._q_values_1, action_indices)
            q_values_selected_2 = tf.gather_nd(self._q_values_2, action_indices)

            self._value_loss_1 = tf.losses.huber_loss(q_values_selected_1, tf.stop_gradient(td_targets))
            self._value_loss_2 = tf.losses.huber_loss(q_values_selected_2, tf.stop_gradient(td_targets))
            self._critic_update_1 = self._get_critic_update(self._critic_1, self._value_loss_1)
            self._critic_update_2 = self._get_critic_update(self._critic_2, self._value_loss_2)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_critic_update_op_1 = self._get_target_critic_update(
                self._target_critic_1,
                self._critic_1
            )
            self._target_critic_update_op_2 = self._get_target_critic_update(
                self._target_critic_2,
                self._critic_2
            )
            self._target_critic_update_op = tf.group([
                self._target_critic_update_op_1,
                self._target_critic_update_op_2
            ])

    # algorithm interface

    def target_network_init(self, sess):
        sess.run(self._targets_init_op)

    def act_batch(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        q_values = sess.run(self._q_values, feed_dict=feed_dict)
        return q_values.tolist()

    def train(self, sess, step_index, batch, critic_update=True):
        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [self._value_loss_1, self._value_loss_2]
        if critic_update:
            ops.append(self._critic_update_1)
            ops.append(self._critic_update_2)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        return {
            'critic lr':  critic_lr,
            'q1 loss': ops_[0],
            'q2 loss': ops_[1]
        }

    def target_critic_update(self, sess):
        sess.run(self._target_critic_update_op)

    def get_weights(self, sess, index=0):
        return {
            'critic_1': self._critic_weights_tool_1.get_weights(sess),
            'critic_2': self._critic_weights_tool_2.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._critic_weights_tool_1.set_weights(sess, weights['critic_1'])
        self._critic_weights_tool_2.set_weights(sess, weights['critic_2'])

    def reset_states(self):
        self._critic_1.reset_states()
        self._critic_2.reset_states()
