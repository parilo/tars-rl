import tensorflow as tf

from .base_algo_discrete import BaseAlgoDiscrete
from .base_algo import network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


class DQN_SAC(BaseAlgoDiscrete):
    def __init__(
        self,
        state_shapes,
        action_size,
        policy,
        critic_q,
        critic_v,
        critic_optimizer,
        actor_optimizer,
        n_step=1,
        critic_grad_val_clip=None,
        critic_grad_norm_clip=None,
        gamma=0.99,
        target_critic_update_rate=1.0,
        h_reward_scale=1.0,
        scope="algorithm",
        placeholders=None,
        actor_lr=None,
        critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        actor_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]}
    ):
        super().__init__(
            state_shapes,
            action_size,
            placeholders,
            critic_optim_schedule,
            training_schedule
        )
        self.actor_lr = actor_lr

        self._policy = policy  # log(p_i)
        self._critic_q = critic_q
        self._critic_v = critic_v
        self._target_critic_v = critic_v.copy(scope=scope + "/target_critic")
        self._policy_wt = ModelWeightsTool(policy)
        # self._critic_q_weights_tool = ModelWeightsTool(critic_q)
        # self._critic_v_weights_tool = ModelWeightsTool(critic_v)
        # self._target_critic_weights_tool = ModelWeightsTool(self._target_critic_v)
        self._critic_optimizer = critic_optimizer
        self._actor_optimizer = actor_optimizer
        self._n_step = n_step
        self._critic_grad_val_clip = critic_grad_val_clip
        self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._target_critic_update_rate = target_critic_update_rate
        self._h_reward_scale = h_reward_scale
        self._actor_optim_schedule = actor_optim_schedule

        with tf.name_scope(scope):
            self.build_graph()

    def _get_critic_update(self, critic, loss, optimizer):
        update_op = network_update(
            loss,
            critic,
            optimizer,
            self._critic_grad_val_clip,
            self._critic_grad_norm_clip
        )
        return update_op

    def _get_target_critic_update(self):
        print('--- self._target_critic_update_rate', self._target_critic_update_rate)
        update_op = target_network_update(
            self._target_critic_v, self._critic_v,
            self._target_critic_update_rate
        )
        return update_op

    def _get_targets_init(self):
        critic_init = target_network_update(
            self._target_critic_v, self._critic_v, 1.0
        )
        return critic_init

    def get_values_of_indices(self, values, indices):
        indices_range = tf.range(tf.shape(indices)[0])
        values_indices = tf.stack([indices_range, indices], axis=1)
        return tf.gather_nd(values, values_indices)

    def get_actor_lr(self, step_index):
        return self.get_schedule_params(self._actor_optim_schedule, step_index)['lr']

    def build_graph(self):
        self.create_placeholders()

        # with tf.name_scope("taking_action"):
        #     self._q_values = self._critic(self.states_ph)

        with tf.name_scope("critic_update"):

            self._policy_logits = self._policy(self.states_ph)
            print('policy_logits', self._policy_logits)
            # stohastic_action = tf.multinomial(tf.nn.softmax(self._policy_logits), 1, output_dtype=tf.int32)[:, 0]
            stohastic_action = self.actions_ph
            print('stohastic_action', stohastic_action)
            stohastic_action_logit = self.get_values_of_indices(
                # tf.log(tf.nn.softmax(self._policy_logits)),
                self._policy_logits,
                stohastic_action
                #tf.multinomial(tf.nn.softmax(self._policy_logits), 1, output_dtype=tf.int32)[:, 0]
            )
            print('stohastic_action_logit', stohastic_action_logit)

            h_reward = - self._h_reward_scale * stohastic_action_logit  # entropy reward
            self._mean_h_reward = tf.reduce_mean(h_reward)
            target_v = self.get_values_of_indices(
                self._critic_q(self.states_ph),
                stohastic_action
            ) + h_reward
            print('target_v', target_v)
            prediction_v = self._critic_v(self.states_ph)[:, 0]
            print('prediction_v', prediction_v)

            gamma = self._gamma ** self._n_step
            target_q = self.rewards_ph + gamma * (1 - self.dones_ph) * self._target_critic_v(self.next_states_ph)[:, 0]
            print('target_q', target_q)
            q_values = self._critic_q(self.states_ph)
            prediction_q = self.get_values_of_indices(
                q_values,
                self.actions_ph
            )
            print('prediction_q', prediction_q)

            self._loss_v = tf.losses.huber_loss(target_v, prediction_v)
            self._loss_q = tf.losses.huber_loss(tf.stop_gradient(target_q), prediction_q)
            # self._loss_pi = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            #     labels=tf.nn.softmax(self._policy_logits),
            #     logits=tf.log(tf.nn.softmax(self._policy_logits) / tf.nn.softmax(q_values))
            # ))
            self._loss_pi = tf.losses.huber_loss(
                prediction_q - prediction_v,
                # prediction_q,
                self.get_values_of_indices(self._policy_logits, self.actions_ph)
            )

            self._critic_v_update = self._get_critic_update(self._critic_v, self._loss_v, self._critic_optimizer)
            self._critic_q_update = self._get_critic_update(self._critic_q, self._loss_q, self._critic_optimizer)
            self._policy_update = self._get_critic_update(self._policy, self._loss_pi, self._actor_optimizer)

        with tf.name_scope("targets_update"):
            self._targets_init_op = self._get_targets_init()
            self._target_critic_update_op = self._get_target_critic_update()


    # algorithm interface

    def target_network_init(self, sess):
        sess.run(self._targets_init_op)

    def act_batch(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        policy_logits = sess.run(self._policy_logits, feed_dict=feed_dict)
        return policy_logits.tolist()

    def train(self, sess, step_index, batch, critic_update=True):

        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.critic_lr_ph: critic_lr,
            self.actor_lr: self.get_actor_lr(step_index),
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}
        }

        ops = [
            self._loss_v,
            self._loss_q,
            self._loss_pi,
            self._mean_h_reward
        ]

        if critic_update:
            ops.extend([
                self._critic_v_update,
                self._critic_q_update
            ])

        ops.append(self._policy_update)

        ops_ = sess.run(ops, feed_dict=feed_dict)
        return {
            'critic lr':  critic_lr,
            'v loss': ops_[0],
            'q loss': ops_[1],
            'pi loss': ops_[2],
            'h r': ops_[3]
        }

    def target_critic_update(self, sess):
        sess.run(self._target_critic_update_op)

    def get_weights(self, sess, index=0):
        return {
            'policy': self._policy_wt.get_weights(sess)
            # 'critic': self._critic_weights_tool.get_weights(sess),
            # 'target_critic': self._target_critic_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._policy_wt.set_weights(sess, weights['policy'])
        # self._critic_weights_tool.set_weights(sess, weights['critic'])
        # self._target_critic_weights_tool.set_weights(sess, weights['target_critic'])

    def reset_states(self):
        self._critic_v.reset_states()
        self._critic_q.reset_states()
        self._policy.reset_states()

    def save_actor(self, sess, path):
        print('-- save actor')
        # vars = self._policy.variables()
        # for v in vars:
        #     print(v)
        # saver = tf.train.Saver(max_to_keep=None, var_list=vars)
        # saver.save(sess, path)
        tf.saved_model.simple_save(
            sess,
            path,
            inputs=dict(zip(
                ['input_' + str(i) for i in range(len(self.states_ph))],
                self.states_ph
            )),
            outputs={
                'actor_output': self._policy_logits
            }
        )
