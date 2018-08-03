import copy
import torch
import torch.nn as nn


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


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
            actor_grad_clip=1.0,
            critic_grad_clip=None,
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
        self._actor_grad_clip = actor_grad_clip
        self._critic_grad_clip = critic_grad_clip
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def act_batch(self, states):
        "returns a batch of actions for a batch of states"
        states = states[0]  # hotfix
        # what about eval?
        with torch.no_grad():
            states = self.to_tensor(states)
            actions = self._actor(states)
            actions = actions.detach().cpu().numpy()
            return actions.tolist()

    def train(self, batch, actor_update=True, critic_update=True):
        "returns loss for a batch of transitions"
        return None

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
        info["algo"] = "base_algo"
        info["actor"] = self._actor.get_info()
        info["critic"] = self._critic.get_info()
        info["n_step"] = self._n_step
        info["actor_grad_clip"] = self._actor_grad_clip
        info["critic_grad_clip"] = self._critic_grad_clip
        info["discount_factor"] = self._gamma
        info["target_actor_update_rate"] = self._target_actor_update_rate
        info["target_critic_update_rate"] = self._target_critic_update_rate
        return info
