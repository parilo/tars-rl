import os

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np

from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.algo.base_algo import BaseAlgo as BaseAlgoAllFrameworks
from rl_server.torch.networks.network_torch import NetworkTorch
from rl_server.server.server_replay_buffer import Transition
from rl_server.server.server_episodes_buffer import ServerEpisodesBuffer


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, num_steps, state_dim, latent_dim, device):
        super(VAE, self).__init__()

        self.device = device

        input_size = num_steps * (state_dim + 2) + 1

        self.e1 = nn.Linear(input_size, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, input_size)

        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.state_dim = state_dim

    def forward(self, trajectory, valid_mask, ep_reward):
        vae_input = t.cat([trajectory, valid_mask], dim=-1)
        vae_input = t.cat([vae_input.view((vae_input.shape[0], -1)), ep_reward], dim=-1)
        z = F.relu(self.e1(vae_input))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = t.exp(log_std)
        z = mean + std * t.FloatTensor(np.random.normal(0, 0.1, size=(std.size()))).to(self.device)

        output_traj, output_valid_size, output_ep_reward = self.decode(z=z)

        return output_traj, output_valid_size, output_ep_reward, mean, std

    def decode(self, z=None, batch_size=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            assert batch_size is not None, 'specify batch_size or z'
            z = t.FloatTensor(
                # np.random.normal(0, 1, size=(batch_size, self.latent_dim)),
                np.random.uniform(-0.2, 0.2, size=(batch_size, self.latent_dim))
            ).to(self.device)#.clamp(-0.5, 0.5)

        traj = F.relu(self.d1(z))
        traj = F.relu(self.d2(traj))
        traj = self.d3(traj)
        traj, ep_reward = traj[:, :-1], traj[:, -2:-1]
        traj = traj.view((z.shape[0], self.num_steps, self.state_dim + 2))
        traj, valid_mask = traj[:, :, :-2], 5 * t.tanh(traj[:, :, -2:])
        return traj, valid_mask, ep_reward


def create_algo(algo_config):
    _, _, state_shapes, action_size = algo_config.get_env_shapes()

    actor_params = get_network_params(algo_config, 'actor')
    actor = NetworkTorch(
        **actor_params,
        device=algo_config.device
    ).to(algo_config.device)

    critic_params = get_network_params(algo_config, 'critic')
    critic_1 = NetworkTorch(
        **critic_params,
        device=algo_config.device
    ).to(algo_config.device)
    critic_2 = NetworkTorch(
        **critic_params,
        device=algo_config.device
    ).to(algo_config.device)

    tanh_limit = algo_config.as_obj()['algorithm']['tanh_limit']
    # num_steps, state_dim, latent_dim, device
    num_steps = 100
    latent_size = 64
    vae = VAE(
        num_steps,
        8,  #state_shapes[0][1],
        latent_size,
        algo_config.device
    ).to(algo_config.device)

    actor_optim_info = algo_config.as_obj()['actor_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    actor_optimizer = get_optimizer_class(actor_optim_info)(actor.parameters(), lr=1)

    critic_optim_info = algo_config.as_obj()['critic_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    critic_optimizer = get_optimizer_class(critic_optim_info)(
        list(critic_1.parameters()) + list(critic_2.parameters()),
        lr=1
    )

    vae_optim_info = algo_config.as_obj()['vae_optim']
    # true value of lr will come from lr_scheduler as multiplicative factor
    vae_optimizer = get_optimizer_class(vae_optim_info)(vae.parameters(), lr=1)

    return VAE_GUIDING(
        state_shapes=state_shapes,
        action_size=action_size,
        num_steps=num_steps,
        latent_size=latent_size,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        vae=vae,
        tanh_limit=tanh_limit,
        n_step=algo_config.algorithm.n_step,
        target_actor_update_rate=algo_config.algorithm.target_actor_update_rate,
        target_critic_update_rate=algo_config.algorithm.target_critic_update_rate,
        actor_optimizer=actor_optimizer,
        actor_optim_schedule=actor_optim_info,
        critic_optimizer=critic_optimizer,
        critic_optim_schedule=critic_optim_info,
        vae_optimizer=vae_optimizer,
        vae_optim_schedule=vae_optim_info,
        training_schedule=algo_config.as_obj()["training"],
        device=algo_config.device
    )


def target_network_update(target_network, source_network, tau):
    for target_param, local_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class VAE_GUIDING(BaseAlgoAllFrameworks):
    def __init__(
        self,
        state_shapes,
        action_size,
        num_steps,
        latent_size,
        actor,
        critic_1,
        critic_2,
        vae,
        tanh_limit,
        actor_optimizer,
        critic_optimizer,
        vae_optimizer,
        n_step=1,
        gamma=0.99,
        target_actor_update_rate=1.0,
        target_critic_update_rate=1.0,
        actor_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        critic_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        vae_optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]},
        device='cpu'
    ):
        super().__init__(
            action_size=action_size,
            actor_optim_schedule=actor_optim_schedule,
            critic_optim_schedule=critic_optim_schedule,
            training_schedule=training_schedule
        )

        self._actor = actor
        self._critic_1 = critic_1
        self._critic_2 = critic_2
        self._vae = vae
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._vae_optimizer = vae_optimizer
        self._n_step = n_step
        self._gamma = gamma
        self._target_actor_update_rate = target_actor_update_rate
        self._target_critic_update_rate = target_critic_update_rate
        self.device = device
        self.tanh_limit = tanh_limit
        self.num_steps = num_steps
        self.latent_size = latent_size
        self.state_shapes = state_shapes

        self._actor_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._actor_optimizer, self.get_actor_lr, last_epoch=-1
        )
        self._critic_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._critic_optimizer, self.get_critic_lr, last_epoch=-1
        )

        self._vae_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self._vae_optimizer, self.get_critic_lr, last_epoch=-1
        )

        from envs.funcs.walker2d import EpisodeVisualizer, calc_trajectory_reward
        self._ep_vis = [EpisodeVisualizer(i) for i in range(5)]
        # self._ep_reward_func = calc_trajectory_reward()

        self._eps_buffer = ServerEpisodesBuffer(10000)
        self._best_traj = None
        self._vae_loss = t.zeros(1)
        self._max_traj_reward = t.zeros(1)
        self._best_traj_r = 0
        self.traj_mean = None
        self.traj_std = None
        self._td_targets = 0

        self.action_noise = dist.multivariate_normal.MultivariateNormal(
            t.zeros(self.action_size).to(self.device),
            covariance_matrix=0.01 * t.eye(self.action_size).to(self.device)
        )

    def _vae_update(self, batch):

        batch_size = len(batch)

        trajectories = np.zeros((batch_size, self.num_steps, self.state_shapes[0][1]), dtype=np.float32)
        valid_masks = np.zeros((batch_size, self.num_steps, 2), dtype=np.int64)
        ep_reward = []

        valid_masks[:, :, 0] = 1
        valid_masks[:, :, 1] = 0

        for traj_i, ep in enumerate(batch):
            states = batch[traj_i][0][0]
            ep_len = len(states)
            if ep_len < self.num_steps:
                trajectories[traj_i, :ep_len] = states
                valid_masks[traj_i, :ep_len, 0] = 0
                valid_masks[traj_i, :ep_len, 1] = 1
            else:
                trajectories[traj_i] = states[:self.num_steps]
                valid_masks[traj_i, :, 0] = 0
                valid_masks[traj_i, :, 1] = 1
            ep_reward.append(np.sum(batch[traj_i][2]))

        # print('--- ep_reward', ep_reward)
        trajectories_t = t.tensor(trajectories[..., :8]).to(self.device)
        valid_masks_t = t.tensor(valid_masks).to(self.device)
        ep_reward_t = 0.01 * t.tensor(np.array(ep_reward, dtype=np.float32).reshape((-1, 1))).to(self.device)
        valid_masks_float = valid_masks_t.float()

        # if self.traj_mean is None:
        #     self.traj_mean = trajectories_t.view(-1, self.state_shapes[0][1]).mean(0)
        #     self.traj_std = trajectories_t.view(-1, self.state_shapes[0][1]).std(0)
        #     print('--- mean', self.traj_mean.shape, self.traj_mean)
        #     print('--- std', self.traj_std.shape, self.traj_std)

        # trajectories_t -= self.traj_mean
        # trajectories_t /= self.traj_std

        output_traj, output_valid_mask, output_ep_reward, mean, std = self._vae(trajectories_t, valid_masks_float, ep_reward_t)

        KL_loss = -(1 + t.log(std) - mean.pow(2) - std).mean()
        self._vae_loss = (
                F.mse_loss(
                    output_traj * valid_masks_float[:, :, 1:2],
                    trajectories_t * valid_masks_float[:, :, 1:2]
                ) +
                F.mse_loss(output_ep_reward, ep_reward_t) +
                F.cross_entropy(
                    output_valid_mask.view(-1, 2),
                    valid_masks_t[:, :, 1].view(-1)
                ) +
                0.5 * KL_loss
        )

        self._vae_optimizer.zero_grad()
        self._vae_loss.backward()
        self._vae_optimizer.step()

        all_test_traj, all_test_valid_mask, test_ep_reward = self._vae.decode(batch_size=500)

        # test_traj *= self.traj_std
        # test_traj += self.traj_mean

        # max_traj_ind = t.argmax(test_ep_reward.view(-1))
        max_traj_ind = np.random.randint(0, 500, 1)
        test_traj, test_valid_mask = all_test_traj[max_traj_ind], all_test_valid_mask[max_traj_ind]

        traj_arr = test_traj.cpu().detach().numpy()
        test_valid_mask_arr = t.argmax(test_valid_mask, dim=-1).cpu().detach().numpy()
        traj_arr = traj_arr[test_valid_mask_arr == 1]

        self._best_traj = t.tensor(traj_arr).to(self.device)
        # self._best_traj = test_traj[max_traj_ind][t.argmax(test_valid_mask, dim=-1) == 1].detach()

        # print('---  traj_arr', traj_arr.shape)
        # print(test_valid_mask_arr)
        # print(max_traj_ind.item(), test_ep_reward.max().item(), test_ep_reward.view(-1))
        self._max_traj_reward = test_ep_reward.max().item()
        self._ep_vis[0].show([[np.concatenate(
            [traj_arr, np.zeros((traj_arr.shape[0], 9))], axis=-1
        )]])

        # for vis_i, traj_i in enumerate(np.random.randint(0, 500, 4).tolist()):
        #     rand_test_traj, rand_test_valid_mask = all_test_traj[traj_i], all_test_valid_mask[traj_i]
        #     rand_traj_arr = rand_test_traj.cpu().detach().numpy()
        #     rand_test_valid_mask_arr = t.argmax(rand_test_valid_mask, dim=-1).cpu().detach().numpy()
        #     rand_traj_arr = rand_traj_arr[rand_test_valid_mask_arr == 1]
        #     self._ep_vis[vis_i].show([[np.concatenate(
        #         [rand_traj_arr, np.zeros((rand_traj_arr.shape[0], 9))], axis=-1
        #     )]])

    def _critic_update(self, batch):
        q1_values = self._critic_1(batch.s + [batch.a])
        q2_values = self._critic_1(batch.s + [batch.a])
        next_actions = (
                self.tanh_limit * self._target_actor(batch.s_) +
                self.action_noise.sample((batch.a.shape[0],))
        ).clamp(
            -self.tanh_limit, self.tanh_limit
        )
        next_q1_values = self._target_critic_1(batch.s_ + [next_actions])
        next_q2_values = self._target_critic_2(batch.s_ + [next_actions])
        next_q_values = t.min(next_q1_values, next_q2_values)

        best_traj_r = None
        if self._best_traj is not None:
            best_traj_len = self._best_traj.shape[0]
            # print('--- batch.ep_indices', batch.ep_indices)
            # print('--- best traj', self._best_traj.shape)
            best_traj_r = []
            for i, ep_ind in enumerate(batch.ep_indices.tolist()):
                # print('--- ep ind', ep_ind)
                if ep_ind < (best_traj_len - 1):
                    # print('--- batch.s', batch.s_[0].shape)
                    # print('--- self._best_traj[ep_ind]', self._best_traj[ep_ind].shape)
                    prev_dist = t.abs(batch.s[0][i, 0, :8] - self._best_traj[ep_ind])
                    next_dist = t.abs(batch.s_[0][i, 0, :8] - self._best_traj[ep_ind + 1])
                    best_traj_r.append((prev_dist - next_dist).mean())
                else:
                    best_traj_r.append(0)
            best_traj_r = t.tensor(best_traj_r).to(self.device).view(-1, 1)
            # print('--- best_traj_r', best_traj_r.mean())
            self._best_traj_r = best_traj_r.mean().item()

        self._r_mean = batch.r[:, None].mean().item()
        td_targets = batch.r[:, None] + self._gamma * (1 - batch.done[:, None]) * next_q_values
        self._td_targets = td_targets.mean().item()
        if best_traj_r is not None:
            # print('--- td_targets', td_targets.shape, best_traj_r.shape)
            td_targets += best_traj_r
        self._value_loss = F.smooth_l1_loss(q1_values, td_targets.detach()) + \
                           F.smooth_l1_loss(q2_values, td_targets.detach())

        self._critic_optimizer.zero_grad()
        self._value_loss.backward()
        self._critic_optimizer.step()

    def _actor_update(self, batch):
        q1_values = self._critic_1(batch.s + [self.tanh_limit * self._actor(batch.s)])
        self._policy_loss = -q1_values.mean()

        self._actor_optimizer.zero_grad()
        self._policy_loss.backward()
        self._actor_optimizer.step()

    def target_network_init(self):
        self._target_actor = self._actor.copy().to(self.device)
        self._target_critic_1 = self._critic_1.copy().to(self.device)
        self._target_critic_2 = self._critic_2.copy().to(self.device)

    def act_batch(self, states):
        with t.no_grad():
            model_input = []
            for state_part in states:
                model_input.append(t.tensor(state_part).to(self.device))
            return self.tanh_limit * self._actor(model_input).cpu().numpy().tolist()

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
            batch.ep_indices
        )

        for _ in range(5):
            if self._eps_buffer.get_stored_in_buffer() > 400:
                self._vae_update(self._eps_buffer.get_batch(256))

        self._critic_update(batch_tensors)
        self._critic_lr_scheduler.step()

        self._actor_update(batch_tensors)
        self._actor_lr_scheduler.step()

        return {
            'critic lr':  self._critic_lr_scheduler.get_lr()[0],
            'actor lr': self._actor_lr_scheduler.get_lr()[0],
            'vae loss': self._vae_loss.item(),
            'max_guide_reward': self._max_traj_reward,
            'best_traj r': self._best_traj_r,
            'r mean': self._r_mean,
            'td_targets r': self._td_targets,
            'q loss': self._value_loss.item(),
            'pi loss': self._policy_loss.item(),
            'eps buffer': self._eps_buffer.get_stored_in_buffer_info()
        }

    def target_actor_update(self):
        target_network_update(self._target_actor, self._actor, self._target_actor_update_rate)

    def target_critic_update(self):
        target_network_update(self._target_critic_1, self._critic_1, self._target_critic_update_rate)
        target_network_update(self._target_critic_2, self._critic_2, self._target_critic_update_rate)

    def get_weights(self, index=0):
        return {
            'actor': self._actor.get_weights(),
        }

    def set_weights(self, weights):
        self._actor.set_weights(weights['actor'])

    def reset_states(self):
        self._actor.reset_states()
        self._critic_1.reset_states()
        self._critic_2.reset_states()

    def is_trains_on_episodes(self):
        return False

    def is_on_policy(self):
        return False

    def _get_model_path(self, dir, index, model):
        return os.path.join(dir, "{}-{}.pt".format(model, index))

    def load(self, dir, index):
        self._actor.load_state_dict(t.load(self._get_model_path(dir, index, 'actor')))
        self._critic_1.load_state_dict(t.load(self._get_model_path(dir, index, 'critic_1')))
        self._critic_2.load_state_dict(t.load(self._get_model_path(dir, index, 'critic_2')))

    def save(self, dir, index):
        t.save(self._actor.state_dict(), self._get_model_path(dir, index, 'actor'))
        t.save(self._critic_1.state_dict(), self._get_model_path(dir, index, 'critic_1'))
        t.save(self._critic_2.state_dict(), self._get_model_path(dir, index, 'critic_2'))

    def process_episode(self, episode):
        if len(episode[1]) > 10:
            self._eps_buffer.push_episode(episode)
