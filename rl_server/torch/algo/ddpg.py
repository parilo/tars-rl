import os

import torch as t
import torch.nn.functional as F
import torch.optim as optim

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition


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
    # true value of lr will come from lr_scheduler as multiplicative factor
    actor_optimizer = get_optimizer_class(actor_optim_info)(actor.parameters(), lr=1)

    critic_optim_info = algo_config.as_obj()['critic_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    critic_optimizer = get_optimizer_class(critic_optim_info)(critic.parameters(), lr=1)

    return DDPG(
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


class DDPG(BaseAlgoAllFrameworks):
    def __init__(
        self,
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
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        # self._actor_grad_val_clip = actor_grad_val_clip
        # self._actor_grad_norm_clip = actor_grad_norm_clip
        # self._critic_grad_val_clip = critic_grad_val_clip
        # self._critic_grad_norm_clip = critic_grad_norm_clip
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

    def _critic_update(self, batch):
        q_values = self._critic(batch.s + [batch.a])
        next_actions = self._actor(batch.s_)
        next_q_values = self._target_critic(batch.s_ + [next_actions])
        gamma = self._gamma ** self._n_step
        td_targets = batch.r[:, None] + gamma * (1 - batch.done[:, None]) * next_q_values
        self._value_loss = F.smooth_l1_loss(q_values, td_targets.detach())

        self._critic_optimizer.zero_grad()
        self._value_loss.backward()
        self._critic_optimizer.step()

    def _actor_update(self, batch):
        q_values = self._critic(batch.s + [self._actor(batch.s)])
        self._policy_loss = -q_values.mean()

        self._actor_optimizer.zero_grad()
        self._policy_loss.backward()
        self._actor_optimizer.step()

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

        if actor_update:
            self._actor_update(batch_tensors)
            self._actor_lr_scheduler.step()

        return {
            'critic lr':  self._critic_lr_scheduler.get_lr()[0],
            'actor lr': self._actor_lr_scheduler.get_lr()[0],
            'q loss': self._value_loss.item(),
            'pi loss': self._policy_loss.item()
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
        self._actor.load_state_dict(t.load(self._get_model_path(dir, index, 'actor')))
        self._critic.load_state_dict(t.load(self._get_model_path(dir, index, 'critic')))

    def save(self, dir, index):
        t.save(self._actor.state_dict(), self._get_model_path(dir, index, 'actor'))
        t.save(self._critic.state_dict(), self._get_model_path(dir, index, 'critic'))
