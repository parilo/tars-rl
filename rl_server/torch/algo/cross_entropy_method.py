import os

import torch as t

# from .base_algo import BaseAlgo, network_update, target_network_update
# from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool
from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
# from rl_server.tensorflow.networks.network_keras import NetworkKeras
# from rl_server.tensorflow.algo.base_algo import create_placeholders

from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch


def create_algo(algo_config):
    # _, _, state_shapes, action_size = algo_config.get_env_shapes()

    actor_params = get_network_params(algo_config, 'actor')
    actor = NetworkTorch(
        # state_shapes=state_shapes,
        # action_size=action_size,
        **actor_params,
    )

    # optimizer = optim.Adam(parameters, lr=opt.lr)
    actor_optim_info = algo_config.as_obj()['actor_optim']
    optimizer = get_optimizer_class(actor_optim_info)(actor.parameters())

    return CEM(
        # state_shapes=state_shapes,
        # action_size=action_size,
        actor=actor,
        optimizer=optimizer,
        optim_schedule=actor_optim_info,
        training_schedule=algo_config.as_obj()["training"]
    )


# def ddpg_create_algo(AlgoClass, algo_config, placeholders, scope_postfix):
#
#     _, _, state_shapes, action_size = algo_config.get_env_shapes()
#     if placeholders is None:
#         placeholders = create_placeholders(state_shapes, action_size)
#     algo_scope = 'ddpg' + scope_postfix
#     actor_lr = placeholders[0]
#     critic_lr = placeholders[1]
#
#     actor_params = get_network_params(algo_config, 'actor')
#     actor = NetworkKeras(
#         state_shapes=state_shapes,
#         action_size=action_size,
#         **actor_params,
#         scope="actor_" + scope_postfix
#     )
#
#     critic_params = get_network_params(algo_config, 'critic')
#     critic = NetworkKeras(
#         state_shapes=state_shapes,
#         action_size=action_size,
#         **critic_params,
#         scope="critic_" + scope_postfix
#     )
#
#     actor_optim_info = algo_config.as_obj()['actor_optim']
#     critic_optim_info = algo_config.as_obj()['critic_optim']
#
#     return DDPG(
#         state_shapes=state_shapes,
#         action_size=action_size,
#         actor=actor,
#         critic=critic,
#         actor_optimizer=get_optimizer_class(actor_optim_info)(
#             learning_rate=actor_lr),
#         critic_optimizer=get_optimizer_class(critic_optim_info)(
#             learning_rate=critic_lr),
#         **algo_config.as_obj()["algorithm"],
#         scope=algo_scope,
#         placeholders=placeholders,
#         actor_optim_schedule=actor_optim_info,
#         critic_optim_schedule=critic_optim_info,
#         training_schedule=algo_config.as_obj()["training"]
#     )


class CEM(BaseAlgoAllFrameworks):
    def __init__(
        self,
        # state_shapes,
        # action_size,
        actor,
        optimizer,
        # actor_grad_val_clip=1.0,
        # actor_grad_norm_clip=None,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]}
    ):

        super().__init__(
            # state_shapes,
            # action_size,
            actor_optim_schedule=optim_schedule,
            training_schedule=training_schedule
        )

        self._actor = actor
        # self._critic = critic
        # self._target_actor = actor.copy(scope=scope + "/target_actor")
        # self._target_critic = critic.copy(scope=scope + "/target_critic")
        # self._actor_weights_tool = ModelWeightsTool(actor)
        # self._critic_weights_tool = ModelWeightsTool(critic)
        self._optimizer = optimizer
        # self._critic_optimizer = critic_optimizer
        # self._n_step = n_step
        # self._actor_grad_val_clip = actor_grad_val_clip
        # self._actor_grad_norm_clip = actor_grad_norm_clip
        # self._critic_grad_val_clip = critic_grad_val_clip
        # self._critic_grad_norm_clip = critic_grad_norm_clip
        # self._gamma = gamma
        # self._target_actor_update_rate = target_actor_update_rate
        # self._target_critic_update_rate = target_critic_update_rate
        #
        # with tf.name_scope(scope):
        #     self.build_graph()
        pass

    # def _get_gradients_wrt_actions(self):
    #     q_values = self._critic(self.states_ph + [self.actions_ph])
    #     gradients = tf.gradients(q_values, self.actions_ph)[0]
    #     return gradients
    #
    # def _get_actor_update(self, loss):
    #     update_op = network_update(
    #         loss, self._actor, self._actor_optimizer,
    #         self._actor_grad_val_clip, self._actor_grad_norm_clip)
    #     return update_op
    #
    # def _get_critic_update(self, loss):
    #     update_op = network_update(
    #         loss, self._critic, self._critic_optimizer,
    #         self._critic_grad_val_clip, self._critic_grad_norm_clip)
    #     return update_op
    #
    # def _get_target_actor_update(self):
    #     update_op = target_network_update(
    #         self._target_actor, self._actor,
    #         self._target_actor_update_rate)
    #     return update_op
    #
    # def _get_target_critic_update(self):
    #     update_op = target_network_update(
    #         self._target_critic, self._critic,
    #         self._target_critic_update_rate)
    #     return update_op
    #
    # def _get_targets_init(self):
    #     actor_init = target_network_update(
    #         self._target_actor, self._actor, 1.0)
    #     critic_init = target_network_update(
    #         self._target_critic, self._critic, 1.0)
    #     return tf.group(actor_init, critic_init)
    #
    # def build_graph(self):
    #     self.create_placeholders()
    #
    #     with tf.name_scope("taking_action"):
    #         self._actions = self._actor(self.states_ph)
    #         self._gradients = self._get_gradients_wrt_actions()
    #
    #     with tf.name_scope("actor_update"):
    #         self._q_values = self._critic(self.states_ph + [self._actor(self.states_ph)])
    #         self._policy_loss = -tf.reduce_mean(self._q_values)
    #         self._actor_update = self._get_actor_update(self._policy_loss)
    #
    #     with tf.name_scope("critic_update"):
    #         q_values = self._critic(self.states_ph + [self.actions_ph])
    #         next_actions = self._actor(self.next_states_ph)
    #         # next_actions = self._target_actor(self.next_states_ph)
    #         next_q_values = self._target_critic(
    #             self.next_states_ph + [next_actions])
    #         # print(self._gamma, self._n_step)
    #         gamma = self._gamma ** self._n_step
    #         td_targets = self.rewards_ph[:, None] + gamma * (
    #             1 - self.dones_ph[:, None]) * next_q_values
    #         self._value_loss = tf.losses.huber_loss(
    #             q_values, tf.stop_gradient(td_targets))
    #         self._critic_update = self._get_critic_update(self._value_loss)
    #
    #     with tf.name_scope("targets_update"):
    #         self._targets_init_op = self._get_targets_init()
    #         self._target_actor_update_op = self._get_target_actor_update()
    #         self._target_critic_update_op = self._get_target_critic_update()

    def target_network_init(self):
        # sess.run(self._targets_init_op)
        pass

    def act_batch(self, states):
        # feed_dict = dict(zip(self.states_ph, states))
        # actions = sess.run(self._actions, feed_dict=feed_dict)
        # return actions.tolist()
        pass

    def act_batch_with_gradients(self, states):
        # feed_dict = dict(zip(self.states_ph, states))
        # actions = sess.run(self._actions, feed_dict=feed_dict)
        # feed_dict = {**feed_dict, **{self.actions_ph: actions}}
        # gradients = sess.run(self._gradients, feed_dict=feed_dict)
        # return actions.tolist(), gradients.tolist()
        pass

    def train(self, step_index, batch, actor_update=True, critic_update=True):
        # actor_lr = self.get_actor_lr(step_index)
        # critic_lr = self.get_critic_lr(step_index)
        # feed_dict = {
        #     self.actor_lr_ph: actor_lr,
        #     self.critic_lr_ph: critic_lr,
        #     **dict(zip(self.states_ph, batch.s)),
        #     **{self.actions_ph: batch.a},
        #     **{self.rewards_ph: batch.r},
        #     **dict(zip(self.next_states_ph, batch.s_)),
        #     **{self.dones_ph: batch.done}}
        # ops = [self._value_loss, self._policy_loss]
        # if critic_update:
        #     ops.append(self._critic_update)
        # if actor_update:
        #     ops.append(self._actor_update)
        # ops_ = sess.run(ops, feed_dict=feed_dict)
        # return {
        #     'critic lr':  critic_lr,
        #     'actor lr': actor_lr,
        #     'q loss': ops_[0],
        #     'pi loss': ops_[1]
        # }
        pass

    def target_actor_update(self):
        # sess.run(self._target_actor_update_op)
        pass

    def target_critic_update(self):
        # sess.run(self._target_critic_update_op)
        pass

    def get_weights(self, index=0):
        # return {
        #     'actor': self._actor_weights_tool.get_weights(sess),
        #     'critic': self._critic_weights_tool.get_weights(sess)
        # }
        pass

    def set_weights(self, weights):
        # self._actor_weights_tool.set_weights(sess, weights['actor'])
        # self._critic_weights_tool.set_weights(sess, weights['critic'])
        pass

    def reset_states(self):
        pass

    def _get_model_path(self, dir, index):
        return os.path.join(dir, "actor-{}.pt".format(index))

    def load(self, dir, index):
        path = self._get_model_path(dir, index)
        self._actor.load_state_dict(t.load(path))

    def save(self, dir, index):
        path = self._get_model_path(dir, index)
        t.save(self._actor.state_dict(), path)