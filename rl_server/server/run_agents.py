#!/usr/bin/env python

import copy

from multiprocessing import Process
import atexit
import time

from misc.common import parse_run_agents_args
from misc.config import load_config
from rl_server.server.agent import run_agent


args = parse_run_agents_args()

config = load_config(args.config)

ps = []

agent_id = 0


def set_default(obj, param_name, value):
    if param_name not in obj:
        obj[param_name] = value


for algorithm_agents_info in config.as_obj()['agents']:

    if config.is_ensemble():
        algorithm_id = algorithm_agents_info['algorithm_id']
        algo_config = config.get_algo_config(algorithm_id)
    else:
        algorithm_id = 0
        algo_config = config

    for agents_config in algorithm_agents_info['agents']:
        for _ in range(int(agents_config['agents_count'])):
            agent_config = copy.deepcopy(agents_config)
            agent_config['algorithm_id'] = algorithm_id

            del agent_config['agents_count']

            set_default(agent_config, 'agent_id', agent_id)
            set_default(agent_config, 'visualize', False)
            set_default(agent_config, 'exploration', None)
            set_default(agent_config, 'store_episodes', False)
            set_default(agent_config, 'seed', agent_id)

            print(agent_config)
            p = Process(target=run_agent, args=(algo_config, agent_config))
            p.start()
            ps.append(p)

            agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)
