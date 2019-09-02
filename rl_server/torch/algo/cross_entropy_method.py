import os

import torch as t
import torch.distributions as dist
import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch


def create_algo(algo_config):
    _, _, state_shapes, action_size = algo_config.get_env_shapes()

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
        action_size=action_size,
        actor=actor,
        optimizer=optimizer,
        optim_schedule=actor_optim_info,
        training_schedule=algo_config.as_obj()["training"],
        device=algo_config.device
    )


class CEM(BaseAlgoAllFrameworks):
    def __init__(
        self,
        # state_shapes,
        action_size,
        actor,
        optimizer,
        # actor_grad_val_clip=1.0,
        # actor_grad_norm_clip=None,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu'
    ):

        super().__init__(
            # state_shapes,
            action_size=action_size,
            actor_optim_schedule=optim_schedule,
            training_schedule=training_schedule
        )

        self._actor = actor
        self._optimizer = optimizer
        self.device = device

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

    def _postprocess_actions(self, actions):
        print('--- asd', self.action_size, actions[:, :self.action_size], actions[:, self.action_size:], t.eye(self.action_size))
        distribution = dist.multivariate_normal.MultivariateNormal(
            actions[:, self.action_size],
            covariance_matrix=t.abs(actions[:, self.action_size:]) * t.eye(self.action_size)
        )
        return distribution.sample()

    def target_network_init(self):
        pass

    def act_batch(self, states):
        with t.no_grad():
            model_input = []
            for state_part in states:
                model_input.append(t.tensor(state_part).to(self.device))
                print('--- input shape:', model_input[-1].shape)
            return self._postprocess_actions(
                self._actor(model_input)
            ).cpu().numpy().tolist()

    def act_batch_with_gradients(self, states):
        print('--- act_batch_with_gradients')
        # feed_dict = dict(zip(self.states_ph, states))
        # actions = sess.run(self._actions, feed_dict=feed_dict)
        # feed_dict = {**feed_dict, **{self.actions_ph: actions}}
        # gradients = sess.run(self._gradients, feed_dict=feed_dict)
        # return actions.tolist(), gradients.tolist()
        pass

    def train(self, step_index, batch_of_episodes):
        print('--- train')

        # episode = [observations, actions, rewards, dones]

        ep_rewards = []
        for episode in batch_of_episodes:
            ep_rewards.append(np.sum(episode[2]))
        ep_rewards = np.array(ep_rewards)

        ep_sorted_by_rewards = reversed(np.argsort(ep_rewards).tolist())

        return {
            'elite r', np.mean(ep_rewards[ep_sorted_by_rewards[:4]]),
            'mean  r', np.mean(ep_rewards)
        }

    def target_actor_update(self):
        pass

    def target_critic_update(self):
        pass

    def get_weights(self, index=0):
        weights = {}
        for name, value in self._actor.state_dict().items():
            print('--- {}: {}'.format(name, value.shape))
            weights[name] = value.cpu().detach().numpy()
        return {
            'actor': weights
        }

    def set_weights(self, weights):
        state_dict = {}
        for name, value in weights['actor'].items():
            state_dict[name] = t.tensor(value).to(self.device)
        self._actor.load_state_dict(state_dict)

    def reset_states(self):
        self._actor.reset_states()
        pass

    def _get_model_path(self, dir, index):
        return os.path.join(dir, "actor-{}.pt".format(index))

    def load(self, dir, index):
        print('--- load')
        path = self._get_model_path(dir, index)
        self._actor.load_state_dict(t.load(path))

    def save(self, dir, index):
        print('--- save')
        path = self._get_model_path(dir, index)
        t.save(self._actor.state_dict(), path)

    def is_trains_on_episodes(self):
        return True

    def is_on_policy(self):
        return True