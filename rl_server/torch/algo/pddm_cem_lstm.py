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

    return PDDM_CEM_LSTM(
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


class EpisodeBatchTransitions:
    def __init__(self, ep_batch):

        # eps_len = [len(ep[3]) for ep in ep_batch]
        # print('--- eps_len', eps_len)

        self._ep_batch = ep_batch
        self._batch_size = len(ep_batch)

        self._obs_ind = 0
        self._action_ind = 1
        self._done_ind = 3

        self._ep_is_not_ended = np.ones((self._batch_size,), dtype=np.uint8)
        self._step_i = 0

    def get_ongoing_ep_indices(self):
        return np.array(np.argwhere(self._ep_is_not_ended == 1), dtype=np.int32).reshape((-1,))

    @staticmethod
    def _get_not_ended_eps_indices(ep_is_not_ended):
        return np.argwhere(ep_is_not_ended == 1)

    @staticmethod
    def _collect_step_values_from_eps(ep_batch, ep_component_ind, step_i, ep_is_not_ended, comp_is_list=False):
        if comp_is_list:
            comp_list_size = len(ep_batch[0][0])

        ep_indices = []
        step_vals = [[] * comp_list_size] if comp_is_list else []
        for ep_i in np.argwhere(ep_is_not_ended == 1).tolist():
            ep_i = ep_i[0]
            ep_indices.append(ep_i)
            # print(f'--- ep_i {ep_i} ep_component_ind {ep_component_ind} step_i {step_i}')
            if comp_is_list:
                # print('--- comp', ep_batch[ep_i][ep_component_ind])
                for comp_list_i in range(comp_list_size):
                    step_vals[comp_list_i].append(ep_batch[ep_i][ep_component_ind][comp_list_i][step_i])
            else:
                step_vals.append(ep_batch[ep_i][ep_component_ind][step_i])

        if comp_is_list:
            step_vals = [np.array(step_vals_part) for step_vals_part in step_vals]
        else:
            step_vals = np.array(step_vals)

        return step_vals, np.array(ep_indices, dtype=np.uint32)

    def get_next_step(self):
        ep_not_ended_count = len(EpisodeBatchTransitions._get_not_ended_eps_indices(self._ep_is_not_ended))

        obs, _ = EpisodeBatchTransitions._collect_step_values_from_eps(
            self._ep_batch,
            self._obs_ind,
            self._step_i,
            self._ep_is_not_ended,
            comp_is_list=True
        )

        actions, _ = EpisodeBatchTransitions._collect_step_values_from_eps(
            self._ep_batch,
            self._action_ind,
            self._step_i,
            self._ep_is_not_ended
        )

        # update ongoing episodes indices
        step_done, ep_indices = EpisodeBatchTransitions._collect_step_values_from_eps(
            self._ep_batch,
            self._done_ind,
            self._step_i,
            self._ep_is_not_ended
        )
        step_done = step_done.astype(np.uint8)
        self._ep_is_not_ended[ep_indices] *= (1 - step_done)

        self._step_i += 1

        return obs, actions, ep_not_ended_count, step_done


class PDDM_CEM_LSTM(BaseAlgoAllFrameworks):
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
        self._lstm_hidden_size = 256

        env_funcs_module = importlib.import_module(env_funcs_module)
        self._is_done_func = getattr(env_funcs_module, is_done_func)
        self._calc_reward_func = getattr(env_funcs_module, calc_reward_func)

        self._env_agent_buffers = [AgentBuffer(
            self._num_env_steps + 1,
            self.observation_shapes,
            self.observation_dtypes,
            action_size
        ) for _ in range(self._num_gen_episodes)]

        from envs.funcs.walker2d import EpisodeVisualizer
        self._ep_vis = EpisodeVisualizer()

    def _env_model_update(self, ep_batch):

        batch_size = len(ep_batch)
        ep_transitions = EpisodeBatchTransitions(ep_batch)

        obs_predicted = []
        d_obs_predicted = []
        obs_gt = []
        prev_obs_gt = []

        lstm_hidden = (
            t.zeros(2, batch_size, self._lstm_hidden_size).to(self.device),  # two stacked lstms
            t.zeros(2, batch_size, self._lstm_hidden_size).to(self.device)
        )

        # if 0
        step_done = np.zeros((batch_size,), dtype=np.uint8)
        done_indices = np.argwhere(step_done == 0).reshape((-1,))

        while True:
            # ongoing_ep_indices = ep_transitions.get_ongoing_ep_indices()
            lstm_hidden = (
                lstm_hidden[0][:, done_indices],
                lstm_hidden[1][:, done_indices]
            )

            obs, actions, ep_not_ended_count, step_done = ep_transitions.get_next_step()
            done_indices = np.argwhere(step_done == 0).reshape((-1,))
            # print('--- step done', step_done.shape, lstm_hidden[0].shape, lstm_hidden[1].shape)
            obs[0] = t.tensor(obs[0]).unsqueeze(dim=1).to(self.device)
            # print('--- obs gt', obs[0].shape)
            obs_gt.append(obs[0])

            if ep_not_ended_count < batch_size / 2:
                break

            actions = t.tensor(actions).unsqueeze(dim=1).to(self.device)

            # print('--- step', obs[0].shape, actions.shape, ep_not_ended_count)
            input_obs = obs if len(obs_predicted) == 0 else [obs_predicted[-1]]
            # obs_pred, lstm_hidden = self._env_model(input_obs + [actions], hidden_state=lstm_hidden)
            d_obs_pred, lstm_hidden = self._env_model(input_obs + [actions], hidden_state=lstm_hidden)
            obs_pred = input_obs[0] + d_obs_pred
            lstm_hidden = lstm_hidden[0]

            # print('--- next_obs_pred', next_obs_pred.shape, lstm_hidden[0].shape, lstm_hidden[1].shape)
            prev_obs_gt.append(input_obs[0][done_indices])
            obs_predicted.append(obs_pred[done_indices])
            d_obs_predicted.append(d_obs_pred[done_indices])

        # print('--- train obs', len(obs_gt), len(obs_predicted))
        diff = []
        diff_sqr = []
        for obs_prev_gt_item, obs_gt_item, d_obs_predicted_item in zip(prev_obs_gt, obs_gt[1:], d_obs_predicted):
            # print('--- diff', obs_gt_item.shape, obs_predicted_item.shape)
            diff_step = (obs_gt_item - obs_prev_gt_item) - d_obs_predicted_item
            diff.append(t.abs(diff_step).mean().item())
            diff_sqr.append(diff_step * diff_step)

        # print('--- step diff', diff)
        self._env_model_loss = t.mean(t.cat(diff_sqr, dim=0))

        self._optimizer.zero_grad()
        self._env_model_loss.backward()
        self._optimizer.step()

        train_info = {
            'lr':  self._lr_scheduler.get_lr()[0],
            'env model loss': self._env_model_loss.item()
        }
        train_info.update({f'step_{i}': diff_val for i, diff_val in enumerate(diff)})
        return train_info

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
                lstm_hidden = (
                    t.zeros(2, self._num_gen_episodes, self._lstm_hidden_size).to(self.device),  # two stacked lstms
                    t.zeros(2, self._num_gen_episodes, self._lstm_hidden_size).to(self.device)
                )
                for i in range(self._num_env_steps):
                    action = self._inner_algo.act_batch_tensor(state_tensors)
                    # state_tensors, lstm_hidden = self._env_model(state_tensors + [action.unsqueeze(1)], hidden_state=lstm_hidden)  # states must be list of modalities
                    dstate_tensors, lstm_hidden = self._env_model(state_tensors + [action.unsqueeze(1)], hidden_state=lstm_hidden)  # states must be list of modalities
                    state_tensors = [state_tensors[0] + dstate_tensors]
                    lstm_hidden = lstm_hidden[0]
                    # state_tensors = [state_tensors]
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
                        lstm_hidden[0][:, buf_i] = t.zeros((2, self._lstm_hidden_size))
                        lstm_hidden[1][:, buf_i] = t.zeros((2, self._lstm_hidden_size))

                    # if len(complete_episodes) > self._num_gen_episodes:
                    #     break

                print(f'--- end generation of {i} steps')

        for buf in self._env_agent_buffers:
            if buf.get_episode_len() > 0:
                complete_episodes.append(buf.get_complete_episode())

        print('--- complete episodes', len(complete_episodes))
        if len(complete_episodes) > 0:
            train_info = self._inner_algo.train(step_index, complete_episodes)
            self._ep_vis.show(self._inner_algo.top_1_episode)
            return train_info
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
        # batch_tensors = Transition(
        #     [t.tensor(s).to(self.device) for s in batch.s],
        #     t.tensor(batch.a).to(self.device),
        #     t.tensor(batch.r).to(self.device),
        #     [t.tensor(s_).to(self.device) for s_ in batch.s_],
        #     t.tensor(batch.done).float().to(self.device),
        #     [t.tensor(mask).to(self.device) for mask in batch.valid_mask],
        #     [t.tensor(mask).to(self.device) for mask in batch.next_valid_mask],
        # )

        train_info = self._env_model_update(batch)
        if step_index > self._env_train_start_step_count and step_index % self._update_inner_algo_every == 0:
            for _ in range(2):
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
        return True

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
        self._start_states_buffer.push_episode(episode)
