import os
import random
import numpy as np
import argparse


def set_global_seeds(i):
    # try:
    #     import torch
    # except ImportError:
    #     pass
    # else:
    #     torch.manual_seed(i)
    #     torch.cuda.manual_seed_all(i)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    random.seed(i)
    np.random.seed(i)


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
