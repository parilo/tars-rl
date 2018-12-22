import os
import random
import numpy as np
import argparse


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
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
        "--visualize",
        dest="visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--validation",
        dest="validation",
        action="store_true",
        default=False)
    parser.add_argument(
        "--config",
        type=str, required=True)
    parser.add_argument(
        "--logdir",
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
    # parser.add_argument(
    #     "--step-limit",
    #     dest="step_limit",
    #     type=int,
    #     default=0)
    return parser.parse_args()


def parse_server_args():
    parser = argparse.ArgumentParser(
        description="RL Server")
    parser.add_argument(
        "--config",
        type=str,
        required=True)
    parser.add_argument(
        "--logdir",
        type=str,
        required=True)
    return parser.parse_args()
