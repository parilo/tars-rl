#!/usr/bin/env python

import copy

from multiprocessing import Process
import atexit
import time

from misc.common import parse_run_agents_args
from misc.config import RunAgentsConfig
from rl_server.server.agent import run_agent


args = parse_run_agents_args()

config = RunAgentsConfig(args.config)

ps = []

agent_id = 0


def set_default(obj, param_name, value):
    if param_name not in obj:
        obj[param_name] = value


for algo_agents_config_obj, exp_config in zip(config.as_obj(), config.get_exp_configs()):
    for agents_config in algo_agents_config_obj['agents']:
        for _ in range(int(agents_config['agents_count'])):
            agent_config = copy.deepcopy(agents_config)

            del agent_config['agents_count']

            set_default(agent_config, 'agent_id', agent_id)
            set_default(agent_config, 'visualize', False)
            set_default(agent_config, 'exploration', None)
            set_default(agent_config, 'store_episodes', False)
            set_default(agent_config, 'seed', agent_id)

            print(agent_config)
            p = Process(target=run_agent, args=(exp_config, agent_config))
            p.start()
            ps.append(p)

            agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)
