import tensorflow as tf

from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool


def create_placeholders(state_shapes, action_size, scope="placeholders"):
    with tf.name_scope(scope):
        states_ph, next_states_ph = [], []
        for i, s in enumerate(state_shapes):
            states_batch_shape = [None] + list(s)
            states_ph.append(tf.placeholder(
                tf.float32, states_batch_shape, "states"+str(i)+"_ph"))
            next_states_ph.append(tf.placeholder(
                tf.float32, states_batch_shape, "next_states"+str(i)+"_ph"))
        actions_ph = tf.placeholder(
            tf.float32, [None, action_size], "actions_ph")
        rewards_ph = tf.placeholder(
            tf.float32, [None, ], "rewards_ph")
        dones_ph = tf.placeholder(
            tf.float32, [None, ], "dones_ph")
            
    return (states_ph, actions_ph, rewards_ph, next_states_ph, dones_ph)


class BaseAlgo:
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
        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic = critic
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
        self._placeholders = placeholders
        
        self._target_actor = actor.copy(scope=scope + "/target_actor")
        self._target_critic = critic.copy(scope=scope + "/target_critic")

    @staticmethod
    def target_network_update(target, source, tau):
        update_ops = tf.group(*(
            w_target.assign_sub(tau * (w_target - w_source))
            for w_source, w_target in zip(
                source.variables(), target.variables())))
        return update_ops

    @staticmethod
    def network_update(loss, network, optimizer,
                       grad_val_clip, grad_norm_clip):
        gradients = optimizer.compute_gradients(
            loss, var_list=network.variables())
        if grad_val_clip is not None:
            gradients = [(tf.clip_by_value(
                grad, -grad_val_clip, grad_val_clip), var)
                for grad, var in gradients]
        if grad_norm_clip is not None:
            gradients = [(tf.clip_by_norm(
                grad, grad_norm_clip), var)
                for grad, var in gradients]
        update_op = optimizer.apply_gradients(gradients)
        return update_op

    def create_placeholders(self):
        if self._placeholders is None:
            self._placeholders = create_placeholders(self._state_shapes, self._action_size)

        self.states_ph = self._placeholders[0]
        self.actions_ph = self._placeholders[1]
        self.rewards_ph = self._placeholders[2]
        self.next_states_ph = self._placeholders[3]
        self.dones_ph = self._placeholders[4]
            
        # with tf.name_scope("placeholders"):
        #     self.states_ph, self.next_states_ph = [], []
        #     for i, s in enumerate(self._state_shapes):
        #         states_batch_shape = [None] + list(s)
        #         self.states_ph.append(tf.placeholder(
        #             tf.float32, states_batch_shape, "states"+str(i)+"_ph"))
        #         self.next_states_ph.append(tf.placeholder(
        #             tf.float32, states_batch_shape, "next_states"+str(i)+"_ph"))
        #     self.actions_ph = tf.placeholder(
        #         tf.float32, [None, self._action_size], "actions_ph")
        #     self.rewards_ph = tf.placeholder(
        #         tf.float32, [None, ], "rewards_ph")
        #     self.dones_ph = tf.placeholder(
        #         tf.float32, [None, ], "dones_ph")

    def get_gradients_wrt_actions(self):
        q_values = self._critic([self.states_ph, self.actions_ph])
        gradients = tf.gradients(q_values, self.actions_ph)[0]
        return gradients

    def get_actor_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._actor, self._actor_optimizer,
            self._actor_grad_val_clip, self._actor_grad_norm_clip)
        return update_op

    def get_critic_update(self, loss):
        update_op = BaseAlgo.network_update(
            loss, self._critic, self._critic_optimizer,
            self._critic_grad_val_clip, self._critic_grad_norm_clip)
        return update_op

    def get_target_actor_update(self):
        update_op = BaseAlgo.target_network_update(
            self._target_actor, self._actor,
            self._target_actor_update_rate)
        return update_op

    def get_target_critic_update(self):
        update_op = BaseAlgo.target_network_update(
            self._target_critic, self._critic,
            self._target_critic_update_rate)
        return update_op

    def get_targets_init(self):
        actor_init = BaseAlgo.target_network_update(
            self._target_actor, self._actor, 1.0)
        critic_init = BaseAlgo.target_network_update(
            self._target_critic, self._critic, 1.0)
        return tf.group(actor_init, critic_init)

    def build_graph(self):
        with tf.name_scope("taking_action"):
            self.actions = self._actor(self.states_ph)
            self.gradients = self.get_gradients_wrt_actions()

        with tf.name_scope("actor_update"):
            self.policy_loss = 0
            self.actor_update = self.get_actor_update(self.policy_loss)

        with tf.name_scope("critic_update"):
            self.value_loss = 0
            self.critic_update = self.get_critic_update(self.value_loss)

        with tf.name_scope("targets_update"):
            self.targets_init_op = self.get_targets_init()
            self.target_actor_update_op = self.get_target_actor_update()
            self.target_critic_update_op = self.get_target_critic_update()
        
    def get_value_loss_op(self):
        return self.value_loss
        
    def get_policy_loss_op(self):
        return self.policy_loss
        
    def get_actor_update_op(self):
        return self.actor_update
        
    def get_critic_update_op(self):
        return self.critic_update

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def act_batch(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self.actions, feed_dict=feed_dict)
        return actions.tolist()

    def act_batch_with_gradients(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self.actions, feed_dict=feed_dict)
        feed_dict = {**feed_dict, **{self.actions_ph: actions}}
        gradients = sess.run(self.gradients, feed_dict=feed_dict)
        return actions.tolist(), gradients.tolist()

    def train(self, sess, batch, actor_update=True, critic_update=True):
        feed_dict = {
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [self.value_loss, self.policy_loss]
        if critic_update:
            ops.append(self.critic_update)
        if actor_update:
            ops.append(self.actor_update)
        ops_ = sess.run(ops, feed_dict=feed_dict)
        value_loss, policy_loss = ops_[:2]
        return value_loss

    def target_actor_update(self, sess):
        sess.run(self.target_actor_update_op)

    def target_critic_update(self, sess):
        sess.run(self.target_critic_update_op)

    def target_network_init(self, sess):
        sess.run(self.targets_init_op)

    def _get_info(self):
        info = {}
        info["algo"] = "base"
        info["actor"] = self._actor.get_info()
        info["critic"] = self._critic.get_info()
        return info

    def get_actor_weights_tool(self):
        return self._actor_weights_tool
        
    def get_critic_weights_tool(self):
        return self._critic_weights_tool

    def get_weights(self, sess, index=0):
        return {
            'actor': self._actor_weights_tool.get_weights(sess),
            'critic': self._critic_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._actor_weights_tool.set_weights(sess, weights['actor'])
        self._critic_weights_tool.set_weights(sess, weights['critic'])
        