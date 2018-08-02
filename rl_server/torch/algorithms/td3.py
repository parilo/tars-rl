import copy
import numpy as np
import torch
import torch.nn as nn
from .base_algo import BaseAlgo


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


class HuberLoss(nn.Module):
    def __init__(self, clip_delta):
        super(HuberLoss, self).__init__()
        self.clip_delta = clip_delta

    def forward(self, y_pred, y_true):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        loss = torch.mean(loss)
        return loss


class TD3(BaseAlgo):
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic,
            critic2,
            actor_optimizer,
            critic_optimizer,
            critic2_optimizer,
            n_step=1,
            action_noise_std=0.2,
            action_noise_clip=0.5,
            actor_grad_clip=1.0,
            critic_grad_clip=None,
            gamma=0.99,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0):
        super(TD3, self).__init__(
            state_shapes, action_size, actor, critic, actor_optimizer,
            critic_optimizer, n_step, actor_grad_clip, critic_grad_clip,
            gamma, target_actor_update_rate, target_critic_update_rate)
        self._action_noise_std = action_noise_std
        self._action_noise_clip = action_noise_clip
        self._critic2 = critic2.to(self._device)
        self._critic2_optimizer = critic2_optimizer
        self._target_critic2 = copy.deepcopy(critic2).to(self._device)
        self._criterion = HuberLoss(1.0)

    def target_critic_update(self):
        soft_update(
            self._target_critic, self._critic,
            self._target_critic_update_rate)
        soft_update(
            self._target_critic2, self._critic2,
            self._target_critic_update_rate)

    def train(self, batch, actor_update=True, critic_update=True):
        states, actions, rewards, next_states, done = (
            list(map(
                lambda x: np.array(x),
                [batch.s[0], batch.a, batch.r,  batch.s_[0], batch.done])))

        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        rewards = self.to_tensor(rewards).unsqueeze(1)
        next_states = self.to_tensor(next_states)
        done = self.to_tensor(done.astype(np.float32)).unsqueeze(1)

        # actor loss
        policy_loss = -torch.mean(
            self._critic(states, self._actor(states)))

        # critic loss
        next_actions = self._target_actor(next_states).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(next_actions), std=self._action_noise_std)
        noise_clip = torch.clamp(
            action_noise, -self._action_noise_clip, self._action_noise_clip)
        next_actions = next_actions + noise_clip

        next_q_values = self._target_critic(next_states, next_actions)
        next_q_values2 = self._target_critic2(next_states, next_actions)
        next_q_values = torch.min(next_q_values, next_q_values2)

        gamma = self._gamma ** self._n_step
        td_target = rewards + (1 - done) * gamma * next_q_values

        q_values = self._critic(states, actions)
        q_values2 = self._critic2(states, actions)

        value_loss = self._criterion(q_values, td_target.detach())
        value_loss2 = self._criterion(q_values2, td_target.detach())

        # actor update
        if actor_update:
            self._actor.zero_grad()
            policy_loss.backward()
            if self._actor_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._actor.parameters(),  self._actor_grad_clip)
            self._actor_optimizer.step()

        # critic update
        if critic_update:
            self._critic.zero_grad()
            value_loss.backward()
            if self._critic_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._critic.parameters(), self._critic_grad_clip)
            self._critic_optimizer.step()

            self._critic2.zero_grad()
            value_loss2.backward()
            if self._critic_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._critic2.parameters(), self._critic_grad_clip)
            self._critic2_optimizer.step()

        # metrics = {
        #     "value_loss": value_loss,
        #     "policy_loss": policy_loss
        # }
        loss = value_loss + value_loss2 + policy_loss
        loss = loss.item()
        return loss

    def _get_info(self):
        info = super(TD3, self)._get_info()
        info['algo'] = 'TD3'
        info['critic2'] = self._critic2.get_info()
        return info
