import numpy as np
import torch
import torch.nn as nn
from .base_algo import BaseAlgo


class WeightedHuberLoss(nn.Module):
    def __init__(self, clip_delta):
        super(WeightedHuberLoss, self).__init__()
        self.clip_delta = clip_delta

    def forward(self, y_pred, y_true, weights):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        loss = torch.mean(loss * weights)
        return loss


class QuantileDDPG(BaseAlgo):
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            n_step=1,
            actor_grad_clip=1.0,
            critic_grad_clip=None,
            gamma=0.99,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0):
        super(QuantileDDPG, self).__init__(
            state_shapes, action_size, actor, critic, actor_optimizer,
            critic_optimizer, n_step, actor_grad_clip, critic_grad_clip,
            gamma, target_actor_update_rate, target_critic_update_rate)
        self._criterion = WeightedHuberLoss(1.0)

        num_atoms = self._critic.n_atoms
        tau_min = 1 / (2 * num_atoms)
        tau_max = 1 - tau_min
        tau = torch.linspace(start=tau_min, end=tau_max, steps=num_atoms)
        self.tau = self.to_tensor(tau)
        self.num_atoms = num_atoms

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
        agent_atoms = self._critic(states, actions)
        next_atoms = self._target_critic(
            next_states,
            self._target_actor(next_states),
        ).detach()

        gamma = self._gamma ** self._n_step
        target_atoms = rewards + (1 - done) * gamma * next_atoms

        atoms_diff = target_atoms[:, None, :] - agent_atoms[:, :, None]
        delta_atoms_diff = atoms_diff.lt(0).to(torch.float32).detach()
        huber_weights = torch.abs(
            self.tau[None, :, None] - delta_atoms_diff) / self.num_atoms
        value_loss = self._criterion(
            agent_atoms[:, :, None], target_atoms[:, None, :], huber_weights)

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

        # metrics = {
        #     "value_loss": value_loss,
        #     "policy_loss": policy_loss
        # }
        loss = value_loss + policy_loss
        loss = loss.item()
        return loss

    def _get_info(self):
        info = super(QuantileDDPG, self)._get_info()
        info['algo'] = 'QuantileDDPG'
        info['num_atoms'] = self._critic.n_atoms
        return info