import os
import importlib

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition
from rl_server.torch.algo.cross_entropy_method import create_algo as create_algo_cem
from rl_server.server.start_states_buffer import StartStatesBuffer
from rl_server.server.agent_replay_buffer import AgentBuffer


class NetworkEnsemble(nn.Module):

    def __init__(self, count, env_model_params, device):
        super().__init__()
        self.count = count
        self._models = [
            NetworkTorch(
                **env_model_params,
                device=device
            ).to(device)
            for _ in range(count)
        ]
        for i, model in enumerate(self._models):
            self.add_module('model_' + str(i), model)

    def forward_train(self, x):
        batch_size = x[0].shape[0]
        model_batch_size = int(batch_size / self.count)
        model_ret = []
        for i, model in enumerate(self._models):
            model_x = []
            for x_part in x:
                model_x.append(x_part[model_batch_size * i: model_batch_size * (i + 1)])
            model_ret.append(model(model_x))
        return t.cat(model_ret, dim=0)

    def forward_eval(self, x):
        model_ret = []
        for i, model in enumerate(self._models):
            model_ret.append(model(x))
        return t.stack(model_ret, dim=0).mean(dim=0)

    def forward(self, x, train):
        if train:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def reset_states(self):
        pass


def create_algo(algo_config):
    observation_shapes, observation_dtypes, state_shapes, action_size = algo_config.get_env_shapes()

    env_model_params = get_network_params(algo_config, 'env_model')
    # env_model = NetworkTorch(
    #     **env_model_params,
    #     device=algo_config.device
    # ).to(algo_config.device)
    env_model = NetworkEnsemble(5, env_model_params, algo_config.device)

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
        inner_algo=create_algo_cem(algo_config),
        **algo_config.as_obj()['eml_algorithm']
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
        update_inner_algo_every,
        num_gen_episodes,
        num_env_steps,
        env_train_start_step_count,
        env_funcs_module,
        is_done_func,
        calc_reward_func,
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

        self._update_inner_algo_every = update_inner_algo_every
        self._num_gen_episodes = num_gen_episodes
        self._num_env_steps = num_env_steps
        self._env_train_start_step_count = env_train_start_step_count

        env_funcs_module = importlib.import_module(env_funcs_module)
        self._is_done_func = getattr(env_funcs_module, is_done_func)
        self._calc_reward_func = getattr(env_funcs_module, calc_reward_func)

        self._env_agent_buffers = [AgentBuffer(
            self._num_env_steps + 1,
            self.observation_shapes,
            self.observation_dtypes,
            action_size
        ) for _ in range(self._num_gen_episodes)]

        self._sigma = t.tensor(0.05).float().to(device)

    def _env_model_update(self, batch):
        rollout_len = batch.s[0].shape[1]

        start_state = [s[:, 0] for s in batch.s]
        state = start_state
        rollout_states = []
        for i in range(rollout_len):
            action = batch.a[:, i]
            predicted_state = [self._env_model(state + [action], train=True)[:, 0]]
            # recurrency
            state = predicted_state
            # no recurrency
            # state = [predicted_state[0].clone().detach()]
            rollout_states.append(predicted_state)

        rollout_states = [s[0] for s in rollout_states]
        rollout_states_tensor = t.stack(rollout_states, dim=1)
        done = batch.done.unsqueeze(-1).unsqueeze(-1)
        dstate = (rollout_states_tensor - batch.s_[0]) * (1 - done)
        dstate2 = dstate * dstate
        # dstate2 = t.abs(dstate)
        dstate3 = dstate2.mean(dim=-1)
        # print('--- dstate prev', start_state[0][0], 'action', batch.a[0, 0])
        # print('--- dstate next', rollout_states_tensor[0, 0])
        print('--- dstate', dstate3.mean(dim=0))
        self._env_model_loss = dstate2.mean()

        # start_state = [s[:, :2] for s in batch.s]
        # state = start_state
        # rollout_states = []
        # for i in range(rollout_len - 2):
        #     action = batch.a[:, i]
        #     next_obs = self._env_model(state + [action])
        #     state = [t.cat([state[0][:, 1:], next_obs], dim=1)]
        #     rollout_states.append(next_obs)
        #
        # # rollout_states = [s[0] for s in rollout_states]
        # # rollout_states_tensor = t.stack(rollout_states, dim=1)
        # rollout_states_tensor = t.cat(rollout_states, dim=1)
        # done = batch.done.unsqueeze(-1).unsqueeze(-1)
        # print('--- shapes', rollout_states_tensor.shape, batch.s_[0][:, 2:].shape)
        # dstate = rollout_states_tensor * (1 - done) - batch.s_[0][:, 2:] * (1 - done)

        self._optimizer.zero_grad()
        self._env_model_loss.backward()
        self._optimizer.step()

        return {
            'lr':  self._lr_scheduler.get_lr()[0],
            'env model loss': self._env_model_loss.item(),
            'loss sqrt': self._env_model_loss.sqrt().item(),
            'rmse': (dstate * dstate).mean().sqrt().item(),
            'mae': t.abs(dstate).mean().item()
        }

    def _get_obs_from_states(self, states):
        return [s[:, -1] for s in states]

    def _get_agent_obs(self, agent_index, obs):
        return [o[agent_index] for o in obs]

    def _get_start_states_and_obs(self, batch_size):
        states = self._start_states_buffer.get_batch(batch_size)
        states = [np.array(s) for s in states]
        obs = self._get_obs_from_states(states)
        return states, obs

    def _inner_algo_update(self, step_index):

        complete_episodes = []

        with t.no_grad():
            if self._start_states_buffer.get_num_in_buffer() >= self._num_gen_episodes:
                print('--- updating inner algo')

                # initializing agents buffers
                states, obs = self._get_start_states_and_obs(self._num_gen_episodes)
                for i, agent_buf in enumerate(self._env_agent_buffers):
                    agent_buf.clear()
                    agent_buf.push_init_observation(self._get_agent_obs(i, obs))

                state_tensors = [t.tensor(s).to(self.device) for s in states]
                for i in range(self._num_env_steps):
                    action = self._inner_algo.act_batch_tensor(state_tensors)
                    # print('--- action', action[0])
                    # print('--- state prev', state_tensors[0][0], 'action', action[0])
                    state_tensors = [self._env_model(state_tensors + [action], train=False)]  # states must be list of modalities
                    # print('--- state next', state_tensors[0][0])

                    action_arr = action.cpu().numpy()
                    states_arrs = [s.cpu().numpy() for s in state_tensors]

                    # check ended episodes
                    is_done = self._is_done_func(self._env_agent_buffers, action_arr, states_arrs)

                    # calc rewards for episodes
                    rewards = self._calc_reward_func(self._env_agent_buffers, action_arr, states_arrs)

                    # store to episodes
                    obs = self._get_obs_from_states(states_arrs)
                    for buf_i, agent_buf in enumerate(self._env_agent_buffers):
                        # next_obs, action, reward, done = transition
                        agent_buf.push_transition([
                            self._get_agent_obs(buf_i, obs),
                            action_arr[buf_i],
                            rewards[buf_i],
                            is_done[buf_i]
                        ])

                    # store ended episodes
                    # and start new episodes
                    for buf_i in np.argwhere(is_done > 0).tolist():
                        buf_i = buf_i[0]
                        # if buf_i == 0:
                        #     print('--- end episode', buf_i)
                        agent_buf = self._env_agent_buffers[buf_i]
                        complete_episodes.append(agent_buf.get_complete_episode())
                        agent_buf.clear()
                        start_states_arr, start_obs = self._get_start_states_and_obs(1)
                        agent_buf.push_init_observation(start_obs[0])
                        for part_id in range(len(start_states_arr)):
                            state_tensors[part_id][buf_i] = t.tensor(start_states_arr[0]).to(self.device)

                    # if len(complete_episodes) > self._num_gen_episodes:
                    #     break

                print(f'--- end generation of {i} steps')

        for buf in self._env_agent_buffers:
            if buf.get_episode_len() > 0:
                complete_episodes.append(buf.get_complete_episode())

        print('--- complete episodes', len(complete_episodes))
        if len(complete_episodes) > 0:
            return self._inner_algo.train(step_index, complete_episodes)
        else:
            return {}

    def target_network_init(self):
        pass

    def act_batch(self, states):
        # return np.zeros((1, 1)) + 0.5
        return self._inner_algo.act_batch(states)

    def act_batch_deterministic(self, states):
        # return np.zeros((1, 1)) + 0.5
        return self._inner_algo.act_batch_deterministic(states)

    def act_batch_with_gradients(self, states):
        raise NotImplemented()

    def train(self, step_index, batch, actor_update=True, critic_update=True):
        batch_tensors = Transition(
            [t.tensor(s).to(self.device) for s in batch.s],
            t.tensor(batch.a).to(self.device),
            t.tensor(batch.r).to(self.device),
            [t.tensor(s_).to(self.device) for s_ in batch.s_],
            t.tensor(batch.done).float().to(self.device),
            [t.tensor(mask).to(self.device) for mask in batch.valid_mask],
            [t.tensor(mask).to(self.device) for mask in batch.next_valid_mask],
        )

        train_info = self._env_model_update(batch_tensors)
        if step_index > self._env_train_start_step_count and step_index % self._update_inner_algo_every == 0:
            for _ in range(100):
                train_info.update(
                    self._inner_algo_update(step_index)
                )

        return train_info

    def target_actor_update(self):
        pass

    def target_critic_update(self):
        pass

    def get_weights(self, index=0):
        return {
            # 'env_model': self._env_model.get_weights(),
            'actor': self._inner_algo.actor.get_weights()
        }

    def set_weights(self, weights):
        # self._env_model.set_weights(weights['env_model'])
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
        self._inner_algo.load(dir, index)

    def save(self, dir, index):
        t.save(self._env_model.state_dict(), self._get_model_path(dir, index, 'env_model'))
        self._inner_algo.save(dir, index)

    def process_episode(self, episode):
        # print('--- process_episode', episode[0][0].shape, episode[1].shape, episode[2].shape, episode[3].shape)
        # ep_rewards = self._inner_algo._calc_episodic_rewards([episode])
        # print('--- real reward', ep_rewards)
        # from envs.pendulum import calc_reward
        # reward2 = 0.
        # obs = episode[0][0]
        # actions = episode[1]
        # for i in range(len(obs)):
        #     state = np.zeros((1, 2, 3))
        #     state[:, -1] = obs[i]
        #     reward2 += calc_reward(None, np.array([actions[i-1]]), [state])
        # print('--- imagine reward', 2 * reward2, '\n')
        self._start_states_buffer.push_episode(episode)
