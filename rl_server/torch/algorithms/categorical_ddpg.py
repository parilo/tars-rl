import numpy as np
import torch
import torch.nn as nn
from .base_algo import BaseAlgo


class CategoricalDDPG(BaseAlgo):
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            values_range=(-10., 10.),
            n_step=1,
            actor_grad_clip=1.0,
            critic_grad_clip=None,
            gamma=0.99,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0):
        super(CategoricalDDPG, self).__init__(
            state_shapes, action_size, actor, critic, actor_optimizer,
            critic_optimizer, n_step, actor_grad_clip, critic_grad_clip,
            gamma, target_actor_update_rate, target_critic_update_rate)

        num_atoms = self._critic.n_atoms
        v_min, v_max = values_range
        delta_z = (v_max - v_min) / (num_atoms - 1)
        z = torch.linspace(start=v_min, end=v_max, steps=num_atoms)

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = delta_z
        self.z = self.to_tensor(z)

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
        q_values = torch.sum(
            self._critic(states, self._actor(states)) * self.z, dim=-1)
        policy_loss = -torch.mean(q_values)

        # critic loss
        agent_probs = self._critic(states, actions)
        next_probs = self._target_critic(
            next_states,
            self._target_actor(next_states).detach(),
        )

        gamma = self._gamma ** self._n_step
        target_atoms = rewards + (1 - done) * gamma * self.z

        tz = torch.clamp(target_atoms, self.v_min, self.v_max)
        tz_z = tz[:, None, :] - self.z[None, :, None]
        tz_z = torch.clamp(
            (1.0 - (torch.abs(tz_z) / self.delta_z)), 0., 1.)
        target_probs = torch.einsum(
            'bij,bj->bi', (tz_z, next_probs)).detach()

        value_loss = -torch.sum(
            target_probs * torch.log(agent_probs + 1e-6))

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
