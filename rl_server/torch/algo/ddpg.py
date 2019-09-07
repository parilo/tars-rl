import os

import torch as t
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition

# import tensorflow as tf
#
# from .base_algo import BaseAlgo, network_update, target_network_update
# from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool
# from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
# from rl_server.tensorflow.networks.network_keras import NetworkKeras
# from rl_server.tensorflow.algo.base_algo import create_placeholders


def create_algo(algo_config):
    _, _, state_shapes, action_size = algo_config.get_env_shapes()

    actor_params = get_network_params(algo_config, 'actor')
    actor = NetworkTorch(
        **actor_params,
        device=algo_config.device
    ).to(algo_config.device)

    critic_params = get_network_params(algo_config, 'critic')
    critic = NetworkTorch(
        **critic_params,
        device=algo_config.device
    ).to(algo_config.device)

    actor_optim_info = algo_config.as_obj()['actor_optim']
    actor_optimizer = get_optimizer_class(actor_optim_info)(actor.parameters())

    critic_optim_info = algo_config.as_obj()['critic_optim']
    critic_optimizer = get_optimizer_class(actor_optim_info)(actor.parameters())

    return DDPG(
        # state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        critic=critic,
        n_step=algo_config.algorithm.n_step,
        target_actor_update_rate=algo_config.algorithm.target_actor_update_rate,
        target_critic_update_rate=algo_config.algorithm.target_critic_update_rate,
        actor_optimizer=actor_optimizer,
        actor_optim_schedule=actor_optim_info,
        critic_optimizer=critic_optimizer,
        critic_optim_schedule=critic_optim_info,
        training_schedule=algo_config.as_obj()["training"],
        device=algo_config.device
    )


def target_network_update(target_network, source_network, tau):
    for target_param, local_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# def create_algo(algo_config, placeholders, scope_postfix):
#     return ddpg_create_algo(DDPG, algo_config, placeholders, scope_postfix)
#
#
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


class DDPG(BaseAlgoAllFrameworks):
    def __init__(
        self,
        # state_shapes,
        action_size,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        n_step=1,
        # actor_grad_val_clip=1.0,
        # actor_grad_norm_clip=None,
        # critic_grad_val_clip=None,
        # critic_grad_norm_clip=None,
        gamma=0.99,
        target_actor_update_rate=1.0,
        target_critic_update_rate=1.0,
        # scope="algorithm",
        # placeholders=None,
        actor_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu'
    ):
        super().__init__(
            # state_shapes,
            action_size=action_size,
            # placeholders,
            actor_optim_schedule=actor_optim_schedule,
            critic_optim_schedule=critic_optim_schedule,
            training_schedule=training_schedule
        )

        self._actor = actor
        self._critic = critic
        # self._actor_weights_tool = ModelWeightsTool(actor)
        # self._critic_weights_tool = ModelWeightsTool(critic)
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        # self._n_step = n_step
        # self._actor_grad_val_clip = actor_grad_val_clip
        # self._actor_grad_norm_clip = actor_grad_norm_clip
        # self._critic_grad_val_clip = critic_grad_val_clip
        # self._critic_grad_norm_clip = critic_grad_norm_clip
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate
        self.device = device
        self._n_step = n_step

        self._critic_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._critic_optimizer, self.get_critic_lr, last_epoch=-1
        )
        self._actor_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._actor_optimizer, self.get_actor_lr, last_epoch=-1
        )

        # with tf.name_scope(scope):
        #     self.build_graph()

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

    def _critic_update(self, batch):
        q_values = self._critic(batch.s + [batch.a])
        next_actions = self._actor(batch.s_)
        next_q_values = self._target_critic(batch.s_ + [next_actions])
        gamma = self._gamma ** self._n_step
        td_targets = batch.r[:, None] + gamma * (1 - batch.done[:, None]) * next_q_values
        self._value_loss = F.smooth_l1_loss(q_values, td_targets.detach())
        self._critic_optimizer.zero_grad()
        self._value_loss.backward()

    def target_network_init(self):
        self._target_actor = self._actor.copy().to(self.device)
        self._target_critic = self._critic.copy().to(self.device)

    def act_batch(self, states):
        with t.no_grad():
            model_input = []
            for state_part in states:
                model_input.append(t.tensor(state_part).to(self.device))
            return self._actor(model_input).cpu().numpy().tolist()

    def act_batch_with_gradients(self, states):
        raise NotImplemented()
        # feed_dict = dict(zip(self.states_ph, states))
        # actions = sess.run(self._actions, feed_dict=feed_dict)
        # feed_dict = {**feed_dict, **{self.actions_ph: actions}}
        # gradients = sess.run(self._gradients, feed_dict=feed_dict)
        # return actions.tolist(), gradients.tolist()

    def train(self, step_index, batch, actor_update=True, critic_update=True):
        batch_tensors = Transition(
            [t.tensor(s).to(self.device) for s in batch.s],
            t.tensor(batch.a).to(self.device),
            t.tensor(batch.r).to(self.device),
            [t.tensor(s_).to(self.device) for s_ in batch.s_],
            t.tensor(batch.done).float().to(self.device)
        )

        if critic_update:
            self._critic_update(batch_tensors)
            self._critic_lr_scheduler.step()
        # actor_lr = self.get_actor_lr(step_index)
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
        return {
            'critic lr':  self._critic_lr_scheduler.get_lr(),
            'actor lr': self._actor_lr_scheduler.get_lr(),
            'q loss': self._value_loss.item(),
            'pi loss': 0
        }

    def target_actor_update(self):
        target_network_update(self._target_actor, self._actor, self._target_actor_update_rate)

    def target_critic_update(self):
        target_network_update(self._target_critic, self._critic, self._target_critic_update_rate)

    def get_weights(self, index=0):
        return {
            'actor': self._actor.get_weights(),
            'critic': self._critic.get_weights(),
        }

    def set_weights(self, weights):
        self._actor.set_weights(weights['actor'])
        self._critic.set_weights(weights['critic'])

    def reset_states(self):
        self._actor.reset_states()
        self._critic.reset_states()

    def is_trains_on_episodes(self):
        return False

    def is_on_policy(self):
        return False

    def _get_model_path(self, dir, index, model):
        return os.path.join(dir, "{}-{}.pt".format(model, index))

    def load(self, dir, index):
        print('--- load')
        self._actor.load_state_dict(t.load(self._get_model_path(dir, index, 'actor')))
        self._critic.load_state_dict(t.load(self._get_model_path(dir, index, 'critic')))

    def save(self, dir, index):
        print('--- save')
        t.save(self._actor.state_dict(), self._get_model_path(dir, index, 'actor'))
        t.save(self._critic.state_dict(), self._get_model_path(dir, index, 'critic'))
