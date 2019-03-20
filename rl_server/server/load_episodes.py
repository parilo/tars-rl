#!/usr/bin/env python

import os
import pickle

import numpy as np
import cv2

from misc.common import parse_load_episodes_args
from misc.config import load_config
# from rl_server.server.agent import run_agent
# from rl_server.server.run_agents import get_algo_and_agent_config

args = parse_load_episodes_args()
config = load_config(args.config)

from rl_server.server.rl_client import RLClient

# def split_observation(obs):
#     obs_parts = []
#     for i in range(3):
#         for j in range(3):
#             obs_parts.append(obs[0, i * 28: (i + 1) * 28, j * 28: (j + 1) * 28])
#     return np.stack(obs_parts, axis=0)
#
# def convert_obs(observations):
#     converted_obs = []
#     for i in range(len(observations)):
#         converted_obs.append(split_observation(observations[i]))
#     return np.array(converted_obs, dtype=observations[0].dtype)

# (84, 84) -> [(84, 84), (28, 28)]
# def convert_obs(observations):
#     converted_obs = [[], []]
#     for i in range(len(observations)):
#         # print(observations[i].shape)
#         # converted_obs.append([
#         #     observations[i],
#         #     np.expand_dims(cv2.resize(observations[i][0], (28, 28)), axis=0)
#         # ])
#         # print(converted_obs[-1][0].shape, converted_obs[-1][1].shape)
#         # converted_obs[0].append(observations[i])
#         converted_obs[0].append(np.expand_dims(cv2.resize(observations[i][0], (64, 64)), axis=0))
#         converted_obs[1].append(np.expand_dims(cv2.resize(observations[i][0], (28, 28)), axis=0))
#     converted_obs[0] = np.array(converted_obs[0])
#     converted_obs[1] = np.array(converted_obs[1])
#     return converted_obs

# demo eps
# def convert_obs(observations):
#     converted_obs = [[], []]
#     for i in range(len(observations[0])):
#         converted_obs[0].append(
#             np.transpose(
#                 cv2.resize(
#                     np.transpose(observations[0][i], (1, 2, 0)),
#                     (84, 84)
#                 ),
#                 (2, 0, 1)
#             )
#         )
#         converted_obs[1].append(
#             np.transpose(
#                 cv2.resize(
#                     np.transpose(observations[0][i], (1, 2, 0)),
#                     (32, 32)
#                 ),
#                 (2, 0, 1)
#             )
#         )
#     converted_obs[0] = np.array(converted_obs[0], dtype=np.uint8)
#     converted_obs[1] = np.array(converted_obs[1], dtype=np.uint8)
#     converted_obs.append(observations[1])
#
#     print(converted_obs[0].shape, converted_obs[1].shape, converted_obs[2].shape)
#     return converted_obs

def convert_reward(rewards):
    # print(np.sum(rewards))
    for i in range(len(rewards)):
        rewards[i] = 3. if rewards[i] > 0.05 else 0.01
    # print(np.sum(rewards))


# def convert_reward_breakout(rewards):
#     for i in range(len(rewards)):
#         rewards[i] *= 10.

rl_client = RLClient(
    port=config.server.client_start_port
)

ep_i = 0
while True:
    fpath = os.path.join(args.eps_dir, 'episode_' + str(ep_i)+'.pkl')
    if os.path.isfile(fpath):
        print('loading', fpath)
        with open(fpath, 'rb') as f:
            episode = pickle.load(f)
            # episode[0] = convert_obs(episode[0])
            # convert_reward(episode[2])
            # convert_reward_breakout(episode[2])
            rl_client.store_episode(episode)


            # debug
            # print(len(episode[2][episode[2] != 0.]))
            # print(len(episode[3][episode[3] == True]))
            # print(len(episode[0][0]), len(episode[1]), len(episode[2]), len(episode[3]))
            #
            # ep_dir = 'ep_' + str(ep_i)
            # os.makedirs(ep_dir, exist_ok=True)
            # images = episode[0][0]
            # print(images.shape)
            # for i in range(len(images)):
            #     # print('img', i)
            #     cv2.imwrite(
            #         ep_dir + '/img_' + str(i) + '.jpg',
            #         cv2.resize(np.transpose(images[i], axes=(1, 2, 0)), (800, 800))
            #     )

    else:
        break

    ep_i += 1
