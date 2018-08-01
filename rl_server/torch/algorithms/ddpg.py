import copy
import numpy as np
import torch
import torch.nn as nn


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


class DDPG:
    def __init__(
            self,
            state_shapes,
            action_size,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            n_step=1,
            gradient_clip=1.0,
            gamma=0.99,
            target_actor_update_rate=1.0,
            target_critic_update_rate=1.0):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor.to(self._device)
        self._critic = critic.to(self._device)
        self._target_actor = copy.deepcopy(actor).to(self._device)
        self._target_critic = copy.deepcopy(critic).to(self._device)
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._grad_clip = gradient_clip
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate
        self._criterion = HuberLoss(1.0)

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def act_batch(self, states):
        states = states[0]  # hotfix
        # what about eval?
        with torch.no_grad():
            states = self.to_tensor(states)
            actions = self._actor(states)
            actions = actions.detach().cpu().numpy()
            return actions.tolist()

    def train(self, batch):
        observations, actions, rewards, next_observations, done = (
            list(map(
                lambda x: np.array(x),
                [batch.s[0], batch.a, batch.r,  batch.s_[0], batch.done])))

        observations = self.to_tensor(observations)
        actions = self.to_tensor(actions)
        rewards = self.to_tensor(rewards).unsqueeze(1)
        next_observations = self.to_tensor(next_observations)
        done = self.to_tensor(done.astype(np.float32)).unsqueeze(1)

        # actor loss
        policy_loss = -torch.mean(
            self._critic(observations, self._actor(observations)))

        # critic loss
        next_qvalues = self._target_critic(
            next_observations,
            self._target_actor(next_observations).detach(),
        )

        gamma = self._gamma ** self._n_step
        expected_qvalues = (1 - done) * gamma * next_qvalues
        td_target = rewards + expected_qvalues

        qvalues = self._critic(observations, actions)
        value_loss = self._criterion(qvalues, td_target.detach())

        # actor update
        self._actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(),  self._grad_clip)
        self._actor_optimizer.step()

        # critic update
        self._critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._critic.parameters(), self._grad_clip)
        self._critic_optimizer.step()

        # metrics = {
        #     "value_loss": value_loss,
        #     "policy_loss": policy_loss
        # }
        loss = value_loss + policy_loss
        loss = loss.item()
        return loss

    def target_actor_update(self):
        soft_update(
            self._target_actor, self._actor,
            self._target_actor_update_rate)

    def target_critic_update(self):
        soft_update(
            self._target_critic, self._critic,
            self._target_critic_update_rate)

    def _get_info(self):
        info = {}
        info['algo'] = 'ddpg'
        info['actor'] = self._actor.get_info()
        info['critic'] = self._critic.get_info()
        info['grad_clip'] = self._grad_clip
        info['discount_factor'] = self._gamma
        info['target_actor_update_rate'] = self._target_actor_update_rate
        info['target_critic_update_rate'] = self._target_critic_update_rate
        return info
