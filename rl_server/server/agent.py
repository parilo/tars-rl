#!/usr/bin/env python

import importlib
import copy

from misc.common import parse_agent_args, dict_to_prop_tree
from misc.config import ExperimentConfig
from rl_server.server.rl_agent_loop import RLAgent


def run_agent(exp_config, agent_config, checkpoint_path=None):

    obs_image_resize_to = None
    if exp_config.env.isset('obs_image_resize_to'):
        obs_image_resize_to = exp_config.env.obs_image_resize_to

    obs_image_to_grayscale = False
    if exp_config.env.isset('obs_image_to_grayscale'):
        obs_image_to_grayscale = exp_config.env.obs_image_to_grayscale

    render_with_cv2 = False
    if exp_config.env.isset('render_with_cv2'):
        render_with_cv2 = exp_config.env.render_with_cv2

    render_with_cv2_resize = False
    if exp_config.env.isset('render_with_cv2_resize'):
        render_with_cv2_resize = exp_config.env.render_with_cv2_resize

    if exp_config.env.is_gym:

        import gym
        from envs.gym_env import GymEnvWrapper

        if exp_config.env.isset('env_class'):
            env_module = importlib.import_module(exp_config.env.env_module)
            EnvClass = getattr(env_module, exp_config.env.env_class)

            if exp_config.env.isset('additional_env_parameters'):
                additional_env_parameters = copy.deepcopy(
                    exp_config.as_obj()['env']['additional_env_parameters']
                )

                if 'id_parameter' in additional_env_parameters:
                    additional_env_parameters[
                        additional_env_parameters['id_parameter']
                    ] = agent_config['agent_id']
                    del additional_env_parameters['id_parameter']

                nested_env = EnvClass(
                    **additional_env_parameters
                )
            else:
                nested_env = EnvClass()
        else:
            nested_env = gym.make(exp_config.env.name)

        env = GymEnvWrapper(
            nested_env,
            seed=agent_config['seed'],
            reward_scale=exp_config.env.reward_scale,
            frame_skip=exp_config.env.frame_skip,
            visualize=agent_config['visualize'],
            reinit_random_action_every=exp_config.env.reinit_random_action_every,
            max_episode_length=exp_config.env.max_episode_length,
            obs_is_image=exp_config.env.obs_is_image,
            obs_image_resize_to=obs_image_resize_to,
            obs_image_to_grayscale=obs_image_to_grayscale,
            agent_id=agent_config['agent_id'],
            render_with_cv2=render_with_cv2,
            render_with_cv2_resize=render_with_cv2_resize
        )

    elif hasattr(exp_config.env, 'env_class'):

        env_module = importlib.import_module(exp_config.env.env_module)
        EnvClass = getattr(env_module, exp_config.env.env_class)

        if hasattr(exp_config.env, 'additional_env_parameters'):
            print(exp_config.as_obj()['env']['additional_env_parameters'])
            env = EnvClass(
                seed=agent_config['seed'],
                reward_scale=exp_config.env.reward_scale,
                frame_skip=exp_config.env.frame_skip,
                visualize=agent_config['visualize'],
                reinit_random_action_every=exp_config.env.reinit_random_action_every,
                max_episode_length=exp_config.env.max_episode_length,
                obs_is_image=exp_config.env.obs_is_image,
                obs_image_resize_to=obs_image_resize_to,
                obs_image_to_grayscale=obs_image_to_grayscale,
                **exp_config.as_obj()['env']['additional_env_parameters']
            )
        else:
            env = EnvClass(
                reward_scale=exp_config.env.reward_scale,
                frame_skip=exp_config.env.frame_skip,
                visualize=agent_config['visualize'],
                reinit_random_action_every=exp_config.env.reinit_random_action_every,
                max_episode_length=exp_config.env.max_episode_length,
                obs_is_image=exp_config.env.obs_is_image,
                obs_image_resize_to=obs_image_resize_to,
                obs_image_to_grayscale=obs_image_to_grayscale
            )

    else:
        raise NotImplementedError(exp_config.env.name)

    agent = RLAgent(
        env=env,
        exp_config=exp_config,
        agent_config=dict_to_prop_tree(agent_config),
        checkpoint_path=checkpoint_path
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
