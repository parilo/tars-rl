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
        device=algo_config.device
    )


class CEM(BaseAlgoAllFrameworks):
    def __init__(
        self,
        observation_shapes,
        observation_dtypes,
        action_size,
        actor,
        optimizer,
        # actor_grad_val_clip=1.0,
        # actor_grad_norm_clip=None,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu'
    ):

        super().__init__(
            action_size=action_size,
            actor_optim_schedule=optim_schedule,
            training_schedule=training_schedule
        )

        self.device = device

        self._actor = actor
        self._optimizer = optimizer
        self._action_size_half = self.action_size // 2  # action is mu and sigma, so half is just length of mu

        self._num_elite_eps = 10
        self._transitions_batch_size = 990
        # self._train_transition_batches = 10
        self._transitions_buffer = TransitionsBuffer(
            10000,
            observation_shapes,
            observation_dtypes,
            action_size,
        )

        # self._log_sigma_exploration_distribution = dist.multivariate_normal.MultivariateNormal(
        #     t.tensor([0] * self._action_size_half).float().to(self.device),
        #     covariance_matrix=0.1 * t.eye(self._action_size_half)
        # )

    def _get_means_and_log_sigmas(self, actions, explore):
        # constant sigma
        # return t.tanh(actions[:, :self.action_size]) * 2, t.log(t.ones_like(actions[:, self.action_size:]))

        means_part = actions[:, :self._action_size_half]
        log_sigma_part = actions[:, self._action_size_half:]

        if explore:
            log_sigma_exploration_distribution = dist.multivariate_normal.MultivariateNormal(
                t.zeros_like(log_sigma_part),
                covariance_matrix=0.1 * t.eye(log_sigma_part.shape[-1])
            )
            # print('--- log_sigma_exploration_distribution', log_sigma_exploration_distribution.sample())

        return (
            t.tanh(means_part) * 2,
            t.tanh(log_sigma_part) * 3 +
            (log_sigma_exploration_distribution.sample() if explore else 0)
        )

    def _sample_actions(self, action_means, action_log_sigmas):
        distribution = dist.multivariate_normal.MultivariateNormal(
            action_means,
            covariance_matrix=t.exp(action_log_sigmas) * t.eye(self._action_size_half)
        )
        return t.cat([t.clamp(distribution.sample(), -2, 2), action_log_sigmas], dim=-1)

    # def _calc_cross_entropy_normal(self, means, log_sigmas, values):
    #     x = (values - means) / t.exp(log_sigmas)
    #     x = x * x
    #     return x  # - log_sigmas  # + const

    def target_network_init(self):
        pass

    def act_batch(self, states):
        with t.no_grad():
            model_input = []
            for state_part in states:
                model_input.append(t.tensor(state_part).to(self.device))
            return self._sample_actions(
                *self._get_means_and_log_sigmas(
                    self._actor(model_input),
                    explore=True
                )
            )
            # return self._sample_actions(
            #     self._actor(model_input)
            # ).cpu().numpy().tolist()

    def act_batch_deterministic(self, states):
        with t.no_grad():
            model_input = []
            for state_part in states:
                model_input.append(t.tensor(state_part).to(self.device))
            means, log_sigmas = self._get_means_and_log_sigmas(self._actor(model_input), explore=False)
            return t.cat([means, log_sigmas], dim=-1).cpu().numpy().tolist()

    def act_batch_with_gradients(self, states):
        raise NotImplemented()

    def train(self, step_index, batch_of_episodes):
        # TODO: good init for NN and stds
        # episode = [observations, actions, rewards, dones]

        # calc episodic rewards
        ep_rewards = []
        for episode in batch_of_episodes:
            ep_rewards.append(np.sum(episode[2]))
        ep_rewards = np.array(ep_rewards)

        # get episodes indexes with respect its episodic rewards
        ep_sorted_by_rewards = list(reversed(np.argsort(ep_rewards).tolist()))
        print('--- eps', ep_rewards[ep_sorted_by_rewards])

        # get top num_elite_eps episodes and put it into replay buffer
        elite_eps_ids = ep_sorted_by_rewards[:self._num_elite_eps]
        self._transitions_buffer.clear()
        for ep_id in elite_eps_ids:
            self._transitions_buffer.push_episode(batch_of_episodes[ep_id])

        # get batch of elite episodes transitions to learn on
        transitions_batch = self._transitions_buffer.get_batch(self._transitions_batch_size, history_len=[2] * len(batch_of_episodes[0][0]))

        # learn
        model_input = []
        for state_part in transitions_batch.s:
            model_input.append(t.tensor(state_part).to(self.device))

        predicted_action_means, predicted_action_log_sigmas = self._get_means_and_log_sigmas(self._actor(model_input), explore=False)
        taken_actions_params = t.tensor(transitions_batch.a).to(self.device)
        taken_actions, taken_log_sigmas = taken_actions_params[:, :self._action_size_half], taken_actions_params[:, self._action_size_half:]  #self._get_means_and_log_sigmas(t.tensor(transitions_batch.a).to(self.device), explore=False)

        print('--- predicted', predicted_action_means[:5].detach().cpu().numpy().tolist(), t.exp(predicted_action_log_sigmas[:5]).detach().cpu().numpy().tolist())
        print('--- taken    ', taken_actions[:5].detach().cpu().numpy().tolist(), t.exp(taken_log_sigmas[:5]).detach().cpu().numpy().tolist())
        cem_loss = F.mse_loss(predicted_action_means, taken_actions) + F.mse_loss(predicted_action_log_sigmas, taken_log_sigmas)
        # + F.mse_loss(t.exp(action_log_sigmas), (action_means - action_values) * 100)

        self._optimizer.zero_grad()
        # cross_ent.backward()
        cem_loss.backward()
        self._optimizer.step()

        return {
            'elite r': np.mean(ep_rewards[elite_eps_ids]),
            'mean r': np.mean(ep_rewards),
            # 'ce': cross_ent.item(),
            'loss': cem_loss.item()
        }

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