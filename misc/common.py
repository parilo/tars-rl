import os
import random
import numpy as np
import argparse


def set_global_seeds(seed, framework):
    if framework == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
    elif framework == 'torch':
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        raise NotImplementedError('framework {} is not supported'.format(framework))
    random.seed(seed)
    np.random.seed(seed)


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_agent_args():
    parser = argparse.ArgumentParser(
        description="Run RL agent")
    parser.add_argument(
        "--id",
        dest="id",
        type=int,
        default=0)
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=0)
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--config",
        type=str, required=True)
    parser.add_argument(
        "--store-episodes",
        dest="store_episodes",
        action="store_true",
        default=False)
    parser.add_argument(
        "--random-start",
        dest="random_start",
        action="store_true",
        default=False)
    return parser.parse_args()


def parse_server_args():
    parser = argparse.ArgumentParser(
        description="RL Server")
    parser.add_argument(
        "--config",
        type=str,
        required=True)
    return parser.parse_args()


def parse_run_agents_args():
    parser = argparse.ArgumentParser(
        description="Run several RL agents")
    parser.add_argument(
        "--config",
        type=str, required=True)
    return parser.parse_args()


def parse_play_args():
    parser = argparse.ArgumentParser(
        description="Play")
    parser.add_argument(
        "--config",
        type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str, required=True)
    parser.add_argument(
        "--algorithm_id",
        type=int, required=False, default=0)
    parser.add_argument(
        "--agent_id",
        type=int, required=False, default=0)
    parser.add_argument(
        "--seed",
        type=int, required=False, default=0)
    return parser.parse_args()


def parse_load_episodes_args():
    parser = argparse.ArgumentParser(
        description="Load stored episodes into replay buffer")
    parser.add_argument(
        "--config",
        type=str, required=True)
    parser.add_argument(
        "--eps_dir",
        type=str, required=True)
    parser.add_argument(
        "--agent_id",
        type=int, required=False, default=0)
    return parser.parse_args()


def parse_save_actor_args():
    parser = argparse.ArgumentParser(
        description="Play")
    parser.add_argument(
        "--config",
        type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str, required=True)
    parser.add_argument(
        "--save_path",
        type=str, required=True)
    parser.add_argument(
        "--algorithm_id",
        type=int, required=False, default=0)
    parser.add_argument(
        "--agent_id",
        type=int, required=False, default=0)
    return parser.parse_args()


class PropertyTree:
    def isset(self, name):
        return hasattr(self, name)


def dict_to_prop_tree(input_value):
    if type(input_value) == dict:
        output_value = PropertyTree()
        for key, value in input_value.items():
            setattr(output_value, key, dict_to_prop_tree(value))
    elif type(input_value) == list:
        output_value = [dict_to_prop_tree(v) for v in input_value]
    else:
        return input_value
    return output_value
