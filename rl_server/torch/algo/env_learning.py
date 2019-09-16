import os

import torch as t
import torch.nn.functional as F
import torch.optim as optim

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition
from rl_server.torch.algo.cross_entropy_method import create_algo as create_algo_cem


import random
from threading import RLock

import numpy as np


class StartStatesBuffer:

    def __init__(
            self,
            capacity,
            observation_shapes,
            observation_dtypes,
            history_len
    ):
        self.size = capacity
        self.num_parts = len(observation_shapes)
        self.obs_shapes = observation_shapes
        self.obs_dtypes = observation_dtypes
        self.history_len = history_len

        self._store_lock = RLock()

        self.clear()

    def clear(self):
        with self._store_lock:
            self.num_in_buffer = 0
            self.stored_in_buffer = 0

            # initialize all np.arrays which store necessary data
            self.observations = []
            for part_id in range(self.num_parts):
                # print(part_id, (self.size,), self.obs_shapes[part_id], self.obs_dtypes[part_id])
                obs = np.empty(
                    (self.size,) + (self.history_len,) + tuple(self.obs_shapes[part_id]),
                    dtype=self.obs_dtypes[part_id]
                )
                print('--- reserved for start states', obs.shape, obs.dtype)
                self.observations.append(obs)

            self.pointer = 0

    def push_episode(self, episode):
        """ episode = [observations, actions, rewards, dones]
            observations = [obs_part_1, ..., obs_part_n]
        """
        if len(episode[1]) == 0:
            print('--- warning: received zero length episode')
            return

        with self._store_lock:
            observations, actions, rewards, dones = episode

            for part_id in range(self.num_parts):
                self.observations[part_id][self.pointer] = np.array(observations[part_id][:self.history_len])

            self.stored_in_buffer += 1
            self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
            self.pointer = (self.pointer + 1) % self.size

    def get_num_in_buffer(self):
        return self.num_in_buffer

    def get_batch(self, batch_size):

        with self._store_lock:

            indices = random.sample(range(self.num_in_buffer), k=batch_size)
            states = []
            for part_id in range(self.num_parts):
                state = [self.observations[part_id][indices[i]] for i in range(batch_size)]
                states.append(state)

            return states


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

        self._start_states_buffer = StartStatesBuffer(
            10000,
            observation_shapes,
            observation_dtypes,
            history_len
        )

    def _env_model_update(self, batch):
        predicted_next_s = self._env_model(batch.s + [batch.a])
        self._env_model_loss = F.smooth_l1_loss(predicted_next_s, batch.s_[0])

        self._optimizer.zero_grad()
        self._env_model_loss.backward()
        self._optimizer.step()

    def _inner_algo_update(self):

        num_gen_episodes = 1000

        if self._start_states_buffer.get_num_in_buffer() >= num_gen_episodes:
            print('--- updating inner algo')
            states = self._start_states_buffer.get_batch(num_gen_episodes)
            while True:
                actions = self._inner_algo.act_batch(states)
                states = self._env_model([states] + [actions])
                # store to episodes
                # check ended episodes
                # calc rewards for episodes
                # store ended episodes
                # start new episodes
                # if have enough ended episodes train inner algo on it

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

    def process_episode(self, episode):
        self._start_states_buffer.push_episode(episode)
