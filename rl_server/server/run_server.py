#!/usr/bin/env python

from rl_server.server.rl_server import RLServer
from rl_server.algo.algo_fabric import create_algorithm
from rl_server.tensorflow.algo.base_algo import create_placeholders
from rl_server.tensorflow.algo.algo_ensemble import AlgoEnsemble
from misc.common import create_if_need, set_global_seeds, parse_server_args
from misc.config import load_config


args = parse_server_args()
exp_config = load_config(args.config)
set_global_seeds(exp_config.server.seed)
create_if_need(exp_config.server.logdir)

if exp_config.is_ensemble():

    _, _, state_shapes, action_size = exp_config.get_env_shapes()

    big_batch_ph = create_placeholders(
        state_shapes,
        action_size
    )

    ensemble_algorithms = [
        create_algorithm(
            exp_config.get_algo_config(i),
            big_batch_ph,
            i
        ) for i in range(len(exp_config.ensemble.algorithms))
    ]
    agent_algorithm = AlgoEnsemble(ensemble_algorithms, big_batch_ph)
else:
    agent_algorithm = create_algorithm(exp_config)

rl_server = RLServer(exp_config, agent_algorithm, exp_config.framework)
if hasattr(exp_config.server, 'load_checkpoint'):
    rl_server.load_weights(exp_config.server.load_checkpoint)
rl_server.start()
