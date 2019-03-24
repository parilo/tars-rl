#!/usr/bin/env python

from misc.common import parse_save_actor_args
from misc.config import load_config
from rl_server.tensorflow.agent_model_tf import AgentModel
# from rl_server.server.agent import run_agent
# from rl_server.server.run_agents import get_algo_and_agent_config

args = parse_save_actor_args()
config = load_config(args.config)

agent_model = AgentModel(config, None)
agent_model.load_checkpoint(args.checkpoint)
agent_model.save_actor(args.save_path)

# algo_config, agent_config = get_algo_and_agent_config(
#     config,
#     args.algorithm_id,
#     args.agent_id,
#     args.seed
# )

# run_agent(
#     config,
#     agent_config,
#     checkpoint_path=args.checkpoint
# )
