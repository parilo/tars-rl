import tensorflow as tf

from .base_algo_discrete import BaseAlgoDiscrete
from .base_algo import network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool
from rl_server.tensorflow.algo.algo_fabric import get_network_params, get_optimizer_class
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

    critic = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **critic_params,
        scope='critic_' + scope_postfix
    )

    return DQN(
        state_shapes=state_shapes,
        action_size=action_size,
        critic=critic,
        critic_optimizer=get_optimizer_class(critic_optim_info)(
            learning_rate=critic_lr),
        **algo_config.as_obj()["algorithm"],
        scope=algo_scope,
        placeholders=placeholders,
        critic_optim_schedule=critic_optim_info,
        training_schedule=algo_config.as_obj()["training"]
    )


class DQN(BaseAlgoDiscrete):
    def __init__(
        self,
        state_shapes,
        action_size,
        critic,
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

        self._critic = critic
        self._target_critic = critic.copy(scope=scope + "/target_critic")
        self._critic_weights_tool = ModelWeightsTool(critic)
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._target_critic_update_rate = target_critic_update_rate

        with tf.name_scope(scope):
            self.build_graph()

    def _get_critic_update(self, loss):
        update_op = network_update(
            loss,
            self._critic,
            self._critic_optimizer,
            self._critic_grad_val_clip,
            self._critic_grad_norm_clip
        )
        return update_op

    def _get_target_critic_update(self):
        update_op = target_network_update(
            self._target_critic, self._critic,
            self._target_critic_update_rate)
        return update_op

    def _get_targets_init(self):
        critic_init = target_network_update(
            self._target_critic, self._critic, 1.0)
        return critic_init

    def get_values_of_indices(self, values, indices):
        indices_range = tf.range(tf.shape(indices)[0])
        values_indices = tf.stack([indices_range, indices], axis=1)
        return tf.gather_nd(values, values_indices)

    def build_graph(self):
        self.create_placeholders()

        with tf.name_scope("taking_action"):
            self._q_values = self._critic(self.states_ph)

        with tf.name_scope("critic_update"):
            # double dqn
            next_q_values_critic_argmax = tf.argmax(self._critic(self.next_states_ph), axis=1, output_type=tf.int32)
            next_q_values_target = self._target_critic(self.next_states_ph)
            next_q_values = self.get_values_of_indices(next_q_values_target, next_q_values_critic_argmax)

            # next_q_values = next_q_values_target()
            gamma = self._gamma ** self._n_step
            # td_targets = self.rewards_ph + gamma * (1 - self.dones_ph) * tf.reduce_max(next_q_values, axis=1)
            td_targets = self.rewards_ph + gamma * (1 - self.dones_ph) * next_q_values

            # indices_range = tf.range(tf.shape(self.actions_ph)[0])
            # action_indices = tf.stack([indices_range, self.actions_ph], axis=1)
            # q_values_selected = tf.gather_nd(self._q_values, action_indices)
            q_values_selected = self.get_values_of_indices(self._q_values, self.actions_ph)

            self._value_loss = tf.losses.huber_loss(q_values_selected, tf.stop_gradient(td_targets))
            self._critic_update = self._get_critic_update(self._value_loss)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_critic_update_op = self._get_target_critic_update()

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
        ops = [self._value_loss]
        if critic_update:
            ops.append(self._critic_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        return {
            'critic lr':  critic_lr,
            'q loss': ops_[0]
        }

    def target_critic_update(self, sess):
        sess.run(self._target_critic_update_op)

    def get_weights(self, sess, index=0):
        return {
            'critic': self._critic_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._critic_weights_tool.set_weights(sess, weights['critic'])

    def reset_states(self):
        self._critic.reset_states()

    def save_actor(self, sess, path):
        print('-- save actor')
        tf.saved_model.simple_save(
            sess,
            path,
            inputs=dict(zip(
                ['input_' + str(i) for i in range(len(self.states_ph))],
                self.states_ph
            )),
            outputs={
                'actor_output': self._q_values
            }
        )
