import os

import torch as t
import torch.nn.functional as F
import torch.optim as optim

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition
from rl_server.torch.algo.cross_entropy_method import create_algo as create_algo_cem


def create_algo(algo_config):
    observation_shapes, observation_dtypes, state_shapes, action_size = algo_config.get_env_shapes()

    env_model_params = get_network_params(algo_config, 'env_model')
    env_model = NetworkTorch(
        **env_model_params,
        device=algo_config.device
    ).to(algo_config.device)

    optim_info = algo_config.as_obj()['optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    optimizer = get_optimizer_class(optim_info)(env_model.parameters(), lr=1)

    return EnvLearning(
        observation_shapes=observation_shapes,
        observation_dtypes=observation_dtypes,
        action_size=action_size,
        history_len=algo_config.as_obj()['env']['history_length'],
        env_model=env_model,
        optimizer=optimizer,
        optim_schedule=optim_info,
        training_schedule=algo_config.as_obj()["training"],
        device=algo_config.device,
        inner_algo=create_algo_cem(algo_config)
    )


def target_network_update(target_network, source_network, tau):
    for target_param, local_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class EnvLearning(BaseAlgoAllFrameworks):
    def __init__(
        self,
        observation_shapes,
        observation_dtypes,
        action_size,
        history_len,
        env_model,
        inner_algo,
        optimizer,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu'
    ):
        super().__init__(
            action_size=action_size,
            critic_optim_schedule=optim_schedule,
            training_schedule=training_schedule
        )

        self.observation_shapes = observation_shapes
        self.observation_dtypes = observation_dtypes
        self.history_len = history_len

        self._env_model = env_model
        self._inner_algo = inner_algo
        self._optimizer = optimizer
        self.device = device

        self._lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._optimizer, self.get_critic_lr, last_epoch=-1
        )

    def _env_model_update(self, batch):
        predicted_next_s = self._env_model(batch.s + [batch.a])
        self._env_model_loss = F.smooth_l1_loss(predicted_next_s, batch.s_[0].view(batch.s_[0].shape[0], -1))

        self._optimizer.zero_grad()
        self._env_model_loss.backward()
        self._optimizer.step()

    def _inner_algo_update(self):

        num_gen_episodes = 1000
        # TODO: use stored real start states
        states = t.zeros(self.observation_shapes[0], dtype=self.observation_dtypes)
        while True:
            actions = self._inner_algo.act_batch(states)
            states = self._env_model([states] + [actions])

    def target_network_init(self):
        pass

    def act_batch(self, states):
        return self._inner_algo.act_batch(states)

    def act_batch_deterministic(self, states):
        return self._inner_algo.act_batch_deterministic(states)

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

        self._env_model_update(batch_tensors)

        return {
            'lr':  self._lr_scheduler.get_lr()[0],
            'env model loss': self._env_model_loss.item(),
        }

    def target_actor_update(self):
        pass

    def target_critic_update(self):
        pass

    def get_weights(self, index=0):
        return {
            'env_model': self._env_model.get_weights(),
            'actor': self._inner_algo.actor.get_weights()
        }

    def set_weights(self, weights):
        self._env_model.set_weights(weights['env_model'])
        self._inner_algo.actor.set_weights(weights['actor'])

    def reset_states(self):
        self._env_model.reset_states()

    def is_trains_on_episodes(self):
        return False

    def is_on_policy(self):
        return False

    def _get_model_path(self, dir, index, model):
        return os.path.join(dir, "{}-{}.pt".format(model, index))

    def load(self, dir, index):
        self._env_model.load_state_dict(t.load(self._get_model_path(dir, index, 'env_model')))

    def save(self, dir, index):
        t.save(self._env_model.state_dict(), self._get_model_path(dir, index, 'env_model'))
