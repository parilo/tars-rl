import os
import json
import yaml
import copy
import random
import numpy as np
import collections


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


def merge_dicts(*dicts):
    """
    Recursive dict merge.
    Instead of updating only top-level keys,
        dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys.
    """
    assert len(dicts) > 1

    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        for k, v in merge_dict.items():
            if (k in dict_ and isinstance(dict_[k], dict)
                    and isinstance(merge_dict[k], collections.Mapping)):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def default_parse_args_hparams(args, unknown_args, hparams):
    for arg in unknown_args:
        arg_name, value = arg.split("=")
        arg_name = arg_name[2:]
        value_content, value_type = value.rsplit(":", 1)

        if "/" in arg_name:
            arg_names = arg_name.split("/")
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))

            hparams_ = hparams
            for arg_name in arg_names[:-1]:
                if arg_name not in hparams_:
                    hparams_[arg_name] = {}

                hparams_ = hparams_[arg_name]

            hparams_[arg_names[-1]] = arg_value
        else:
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))
            args.__setattr__(arg_name, arg_value)
    return args, hparams


def load_ordered_yaml(
        stream,
        Loader=yaml.Loader, object_pairs_hook=collections.OrderedDict):

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def default_parse_fn(args, unknown_args):
    args_ = copy.deepcopy(args)

    # load params
    hparams = {}
    for hparams_path in args_.hparams.split(","):
        with open(hparams_path, "r") as fin:
            if hparams_path.endswith("json"):
                hparams_ = json.load(
                    fin, object_pairs_hook=collections.OrderedDict)
            elif hparams_path.endswith("yml"):
                hparams_ = load_ordered_yaml(fin)
            else:
                raise Exception("Unknown file format")
        hparams = merge_dicts(hparams, hparams_)

    args_, hparams = default_parse_args_hparams(args_, unknown_args, hparams)

    if hasattr(args_, "logdir"):
        with open("{}/hparams.json".format(args_.logdir), "w") as fout:
            json.dump(hparams, fout, indent=2)

    # hack with argparse in json
    training_args = hparams.pop("args", None)
    if training_args is not None:
        for key, value in training_args.items():
            arg_value = getattr(args_, key, None)
            if arg_value is None:
                arg_value = value
            setattr(args_, key, arg_value)

    return args_, hparams


def init_episode_storage(agent_id, logdir):
    path_to_episode_storage = os.path.join(logdir, 'episodes', str(agent_id))
    if not os.path.exists(path_to_episode_storage):
        os.makedirs(path_to_episode_storage)
    path, dirs, files = next(os.walk(path_to_episode_storage))
    return path_to_episode_storage, len(files)
