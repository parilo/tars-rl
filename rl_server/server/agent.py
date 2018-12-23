#!/usr/bin/env python

from misc.common import parse_agent_args, dict_to_prop_tree
from misc.config import ExperimentConfig
from rl_server.server.rl_agent import RLAgent


def run_agent(exp_config, agent_config):

    if exp_config.env.is_gym:

        import gym
        from envs.gym_env import GymEnvWrapper
        env = GymEnvWrapper(
            gym.make(exp_config.env.name),
            reward_scale=exp_config.env.reward_scale,
            frame_skip=exp_config.env.frame_skip,
            visualize=agent_config['visualize'],
            reinit_random_action_every=exp_config.env.reinit_random_action_every
        )

    else:
        raise NotImplementedError(exp_config.env.name)

    agent = RLAgent(
        env=env,
        exp_config=exp_config,
        agent_config=dict_to_prop_tree(agent_config)
    )
    agent.run()


if __name__ == "__main__":

    args = parse_agent_args()
    exp_config = ExperimentConfig(args.config)
    agent_config = {
        'visualize': args.visualize,
        'exploration': None,
        'store_episodes': args.store_episodes,
        'agent_id': args.id,
        'seed': args.id
    }
    run_agent(
        exp_config,
        agent_config
    )
