import os

import torch as t
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import ServerBuffer as TransitionsBuffer


def create_algo(algo_config):
    observation_shapes, observation_dtypes, state_shapes, action_size = algo_config.get_env_shapes()

    actor_params = get_network_params(algo_config, 'actor')
    actor = NetworkTorch(
        **actor_params,
    )

    actor_optim_info = algo_config.as_obj()['actor_optim']
    optimizer = get_optimizer_class(actor_optim_info)(actor.parameters())

    return CEM(
        observation_shapes=observation_shapes,
        observation_dtypes=observation_dtypes,
        action_size=action_size,
        actor=actor,
        optimizer=optimizer,
        optim_schedule=actor_optim_info,
        training_schedule=algo_config.as_obj()["training"],
        device=algo_config.device,
        **algo_config.as_obj()['algorithm']
    )


class CEM(BaseAlgoAllFrameworks):
    def __init__(
        self,
        observation_shapes,
        observation_dtypes,
        action_size,
        actor,
        number_elite_episodes,
        transitions_batch_size,
        transitions_buffer_size,
        sigma,
        optimizer,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu',
        tanh_limit=None
    ):

        super().__init__(
            action_size=action_size,
            actor_optim_schedule=optim_schedule,
            training_schedule=training_schedule
        )

        self.device = device
        self.tanh_limit = tanh_limit
        if tanh_limit:
            self._tanh_limit_min = t.tensor(tanh_limit['min']).float()
            self._tanh_limit_delta = t.tensor(tanh_limit['max'] - tanh_limit['min']).float()

        self._actor = actor
        self._optimizer = optimizer

        self._sigma = t.tensor(sigma).float()
        self._num_elite_eps = number_elite_episodes
        self._transitions_batch_size = transitions_batch_size
        self._transitions_buffer = TransitionsBuffer(
            transitions_buffer_size,
            observation_shapes,
            observation_dtypes,
            action_size,
        )

    def _calc_actor_action(self, model_input):
        actor_output = self._actor(model_input)
        if self.tanh_limit:
            actor_output = (t.tanh(actor_output) * 0.5 + 0.5) * self._tanh_limit_delta + self._tanh_limit_min
        return actor_output

    def _sample_actions(self, action_means):
        distribution = dist.multivariate_normal.MultivariateNormal(
            action_means,
            covariance_matrix=self._sigma * t.eye(self.action_size)
        )

        action = distribution.sample()
        if self.tanh_limit:
            action = t.clamp(action, -2, 2)

        return action

    def _calc_episodic_rewards(self, batch_of_episodes):
        ep_rewards = []
        for episode in batch_of_episodes:
            ep_rewards.append(np.sum(episode[2]))
        return np.array(ep_rewards)

    def _get_elite_episodes_ids(self, ep_rewards):
        # get episodes indexes with respect its episodic rewards
        ep_sorted_by_rewards = list(reversed(np.argsort(ep_rewards).tolist()))

        # get top num_elite_eps episodes and put it into replay buffer
        return ep_sorted_by_rewards[:self._num_elite_eps]

    def _get_transitions_batch_from_elite_episodes(self, elite_eps_ids, batch_of_episodes):

        self._transitions_buffer.clear()
        for ep_id in elite_eps_ids:
            self._transitions_buffer.push_episode(batch_of_episodes[ep_id])

        # get batch of elite episodes transitions to learn on
        return self._transitions_buffer.get_batch(self._transitions_batch_size, history_len=[2] * len(batch_of_episodes[0][0]))

    def _get_model_input(self, state):
        model_input = []
        for state_part in state:
            model_input.append(t.tensor(state_part).to(self.device))
        return model_input

    def train(self, step_index, batch_of_episodes):

        ep_rewards = self._calc_episodic_rewards(batch_of_episodes)
        elite_eps_ids = self._get_elite_episodes_ids(ep_rewards)
        transitions_batch = self._get_transitions_batch_from_elite_episodes(elite_eps_ids, batch_of_episodes)

        # learn using transitions batch
        model_input = self._get_model_input(transitions_batch.s)

        predicted_action_means = self._calc_actor_action(model_input)
        taken_actions = t.tensor(transitions_batch.a).to(self.device)

        cem_loss = F.mse_loss(predicted_action_means, taken_actions)

        self._optimizer.zero_grad()
        cem_loss.backward()
        self._optimizer.step()

        return {
            'elite mean r': np.mean(ep_rewards[elite_eps_ids]),
            'elite std r': np.std(ep_rewards[elite_eps_ids]),
            'mean r': np.mean(ep_rewards),
            'std r': np.std(ep_rewards),
            'loss': cem_loss.item()
        }

    def target_network_init(self):
        pass

    def act_batch(self, states):
        with t.no_grad():
            model_input = self._get_model_input(states)
            return self._sample_actions(self._calc_actor_action(model_input))

    def act_batch_deterministic(self, states):
        with t.no_grad():
            model_input = self._get_model_input(states)
            means = self._calc_actor_action(model_input)
            return means.cpu().numpy().tolist()

    def act_batch_with_gradients(self, states):
        raise NotImplemented()

    def target_actor_update(self):
        pass

    def target_critic_update(self):
        pass

    def get_weights(self, index=0):
        weights = {}
        for name, value in self._actor.state_dict().items():
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

    def _get_model_path(self, dir, index):
        return os.path.join(dir, "actor-{}.pt".format(index))

    def load(self, dir, index):
        path = self._get_model_path(dir, index)
        self._actor.load_state_dict(t.load(path))

    def save(self, dir, index):
        path = self._get_model_path(dir, index)
        t.save(self._actor.state_dict(), path)

    def is_trains_on_episodes(self):
        return True

    def is_on_policy(self):
        return True