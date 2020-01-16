import os
from collections import namedtuple

import torch as t
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch

TransitionsBatch = namedtuple("Transition", ("s", "a", "r", "s_", "done"))


def create_algo(algo_config):
    _, _, state_shapes, action_size = algo_config.get_env_shapes()

    actor_params = get_network_params(algo_config, 'actor')
    actor = NetworkTorch(
        **actor_params,
        device=algo_config.device
    ).to(algo_config.device)

    critic_params = get_network_params(algo_config, 'critic')
    critic1 = NetworkTorch(
        **critic_params,
        device=algo_config.device
    ).to(algo_config.device)
    critic2 = NetworkTorch(
        **critic_params,
        device=algo_config.device
    ).to(algo_config.device)

    actor_optim_info = algo_config.as_obj()['actor_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    actor_optimizer = get_optimizer_class(actor_optim_info)(actor.parameters(), lr=1)

    critic_optim_info = algo_config.as_obj()['critic_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    critic_optimizer = get_optimizer_class(critic_optim_info)(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=1
    )

    if algo_config.algorithm.isset('output_scale'):
        output_scale = t.tensor(algo_config.algorithm.output_scale).float().to(algo_config.device)
    else:
        output_scale = t.tensor(1).float().to(algo_config.device)

    # if algo_config.algorithm.isset('action_l2_loss'):
    #     action_l2_loss = t.tensor(algo_config.algorithm.action_l2_loss).float().to(algo_config.device)
    # else:
    #     action_l2_loss = None

    return SAC(
        action_size=action_size,
        actor=actor,
        critic_q1=critic1,
        critic_q2=critic2,
        gamma=algo_config.algorithm.gamma,
        output_scale=output_scale,
        # action_l2_loss=action_l2_loss,
        log_pi_scale=algo_config.algorithm.log_pi_scale,
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


# Soft Actor Critic
class SAC(BaseAlgoAllFrameworks):
    def __init__(
            self,
            action_size,
            actor,
            critic_q1,
            critic_q2,
            output_scale,
            log_pi_scale,
            # action_l2_loss,
            actor_optimizer,
            critic_optimizer,
            gamma,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0,
            actor_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
            critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
            training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
            device='cpu'
    ):
        super().__init__(
            action_size=action_size,
            actor_optim_schedule=actor_optim_schedule,
            critic_optim_schedule=critic_optim_schedule,
            training_schedule=training_schedule
        )

        self._actor = actor
        self._critic1 = critic_q1
        self._critic2 = critic_q2
        self._output_scale = output_scale
        self._output_scale_arr = output_scale.cpu().numpy()
        self._log_pi_scale = log_pi_scale
        # self._action_l2_loss_weight = action_l2_loss
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate
        self.device = device

        self._critic_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._critic_optimizer, self.get_critic_lr, last_epoch=-1
        )
        self._actor_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._actor_optimizer, self.get_actor_lr, last_epoch=-1
        )

        self._action_scale = self._output_scale[:self.action_size]

    def _clip_action(self, action):
        return t.max(t.min(action, self._action_scale), -self._action_scale)

    def _get_mu_and_log_sigma(self, actor, states):
        actor_output = self._output_scale * actor(states)
        mu = actor_output[:, :self.action_size]
        log_sigma = actor_output[:, self.action_size:]
        return mu, log_sigma

    def _calc_action_stochastic(self, states):
        mu, log_sigma = self._get_mu_and_log_sigma(self._actor, states)
        return self._clip_action(t.normal(mu, t.exp(log_sigma)))

    def _calc_action_deterministic(self, states):
        actor_output = self._actor(states)
        return self._action_scale * actor_output[:, :self.action_size]

    def _calc_action_and_log_pi(self, actor, states):
        mu, log_sigma = self._get_mu_and_log_sigma(actor, states)
        # print('--- mu', mu.shape, 'log_sigma', log_sigma.shape)
        # print(mu, log_sigma)
        # sigma_mtx = t.diag_embed(t.exp(log_sigma))
        # print('--- sigma_mtx', sigma_mtx.shape)
        # print(sigma_mtx)
        # action_dist = dist.multivariate_normal.MultivariateNormal(mu, covariance_matrix=sigma_mtx)
        action_dist = dist.normal.Normal(mu, t.exp(log_sigma))
        action = self._clip_action(action_dist.rsample())
        log_pi = action_dist.log_prob(action)
        return action, log_pi

    def _actor_update(self, batch):

        action, log_pi = self._calc_action_and_log_pi(self._actor, batch.s)
        # action, _ = self._get_mu_and_log_sigma(self._actor, batch.s)
        self._policy_q_loss = -self._critic1(batch.s + [action]).mean()
        self._policy_h_loss = (self._log_pi_scale * log_pi).mean()
        self._policy_loss = self._policy_q_loss + self._policy_h_loss

        self._actor_optimizer.zero_grad()
        self._policy_loss.backward()
        self._actor_optimizer.step()

    def _critic_update(self, batch):
        q1_values = self._critic1(batch.s + [batch.a])
        q2_values = self._critic2(batch.s + [batch.a])
        next_action, next_log_pi = self._calc_action_and_log_pi(self._target_actor, batch.s_)

        next_q1_values = self._target_critic1(batch.s_ + [next_action])
        next_q2_values = self._target_critic2(batch.s_ + [next_action])

        target_min_q = t.min(next_q1_values, next_q2_values)
        target_h = (self._log_pi_scale * next_log_pi).unsqueeze(1)
        td_target = batch.r[:, None] + self._gamma * (1 - batch.done[:, None]) * (target_min_q - target_h)
        self._target_min_q_mean = target_min_q.mean()
        self._target_q_h_mean = target_h.mean()

        self._q1_loss = F.smooth_l1_loss(q1_values, td_target.detach())
        self._q2_loss = F.smooth_l1_loss(q2_values, td_target.detach())
        self._q_loss = self._q1_loss + self._q2_loss

        self._critic_optimizer.zero_grad()
        self._q_loss.backward()
        self._critic_optimizer.step()

    def target_network_init(self):
        self._target_actor = self._actor.copy().to(self.device)
        self._target_critic1 = self._critic1.copy().to(self.device)
        self._target_critic2 = self._critic2.copy().to(self.device)

    def _prepare_actor_input(self, states):
        model_input = []
        for state_part in states:
            model_input.append(t.tensor(state_part).to(self.device))
        return model_input

    def act_batch(self, states):
        with t.no_grad():
            return self._calc_action_stochastic(
                self._prepare_actor_input(states)
            ).cpu().numpy().tolist()

    def act_batch_deterministic(self, states):
        with t.no_grad():
            return self._calc_action_deterministic(
                self._prepare_actor_input(states)
            ).cpu().numpy().tolist()

    def act_batch_with_gradients(self, states):
        raise NotImplemented()

    def train(self, step_index, batch, actor_update=True, critic_update=True):
        batch_tensors = TransitionsBatch(
            [t.tensor(s).to(self.device) for s in batch.s],
            t.tensor(batch.a).to(self.device),
            t.tensor(batch.r).to(self.device),
            [t.tensor(s_).to(self.device) for s_ in batch.s_],
            t.tensor(batch.done).float().to(self.device)
        )

        if critic_update:
            self._critic_update(batch_tensors)
            self._critic_lr_scheduler.step()

        if actor_update:
            self._actor_update(batch_tensors)
            self._actor_lr_scheduler.step()

        train_info = {
            'critic lr': self._critic_lr_scheduler.get_lr()[0],
            'actor lr': self._actor_lr_scheduler.get_lr()[0],
            'pi q loss': self._policy_q_loss.item(),
            'pi h loss': self._policy_h_loss.item(),
            'pi loss': self._policy_loss.item(),
            'target q mean': self._target_min_q_mean.item(),
            'target h mean': self._target_q_h_mean.item(),
            'q1 loss': self._q1_loss.item(),
            'q2 loss': self._q2_loss.item(),
            'q loss': self._q_loss.item(),
        }

        return train_info

    def target_actor_update(self):
        target_network_update(self._target_actor, self._actor, self._target_actor_update_rate)

    def target_critic_update(self):
        target_network_update(self._target_critic1, self._critic1, self._target_critic_update_rate)
        target_network_update(self._target_critic2, self._critic2, self._target_critic_update_rate)

    def get_weights(self, index=0):
        return {
            'actor': self._actor.get_weights(),
            'critic1': self._critic1.get_weights(),
            'critic2': self._critic2.get_weights(),
        }

    def set_weights(self, weights):
        self._actor.set_weights(weights['actor'])
        self._critic1.set_weights(weights['critic1'])
        self._critic2.set_weights(weights['critic2'])

    def reset_states(self):
        self._actor.reset_states()
        self._critic1.reset_states()
        self._critic2.reset_states()

    def is_trains_on_episodes(self):
        return False

    def is_on_policy(self):
        return False

    def _get_model_path(self, dir, index, model):
        return os.path.join(dir, "{}-{}.pt".format(model, index))

    def load(self, dir, index):
        self._actor.load_state_dict(t.load(self._get_model_path(dir, index, 'actor')))
        self._critic1.load_state_dict(t.load(self._get_model_path(dir, index, 'critic1')))
        self._critic2.load_state_dict(t.load(self._get_model_path(dir, index, 'critic2')))
        self._actor_optimizer.load_state_dict(t.load(self._get_model_path(dir, index, 'actor_optim')))
        self._critic_optimizer.load_state_dict(t.load(self._get_model_path(dir, index, 'critic_optim')))

    def save(self, dir, index):
        t.save(self._actor.state_dict(), self._get_model_path(dir, index, 'actor'))
        t.save(self._critic1.state_dict(), self._get_model_path(dir, index, 'critic1'))
        t.save(self._critic2.state_dict(), self._get_model_path(dir, index, 'critic2'))
        t.save(self._actor_optimizer.state_dict(), self._get_model_path(dir, index, 'actor_optim'))
        t.save(self._critic_optimizer.state_dict(), self._get_model_path(dir, index, 'critic_optim'))
