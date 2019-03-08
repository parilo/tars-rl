#!/usr/bin/env python

import os
import pickle

import numpy as np

from misc.common import parse_load_episodes_args
from misc.config import load_config
# from rl_server.server.agent import run_agent
# from rl_server.server.run_agents import get_algo_and_agent_config

args = parse_load_episodes_args()
config = load_config(args.config)

from rl_server.server.rl_client import RLClient

rl_client = RLClient(
    port=config.server.client_start_port
)

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

ep_i = 0
while True:
    fpath = os.path.join(args.eps_dir, 'episode_' + str(ep_i)+'.pkl')
    if os.path.isfile(fpath):
        print('loading', fpath)
        with open(fpath, 'rb') as f:
            episode = pickle.load(f)
            # episode[0][0] = convert_obs(episode[0][0])
            rl_client.store_episode(episode)
    else:
        break

    ep_i += 1
