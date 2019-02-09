import os
import copy
from operator import itemgetter
from shutil import copy2

import yaml

from misc.common import dict_to_prop_tree


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.load(f.read())


class BaseConfig:

    def __init__(self, path=None, from_obj=None, default_param_values={}):
        self._path = path
        if from_obj is None:
            assert os.path.isfile(path), 'config with path: {} not found'.format(path)
            self._load_from_obj(load_yaml(path))
        else:
            self._load_from_obj(from_obj)

        self._default_param_values = default_param_values

    def _load_from_obj(self, obj):
        self._config_as_obj = obj
        self._sort_dict()
        self._config = dict_to_prop_tree(self._config_as_obj)

    def _sort_dict(self):
        pass

    def __getattr__(self, name):
        if hasattr(self._config, name):
            return getattr(self._config, name)
        elif name in self._default_param_values:
            return self._default_param_values[name]
        else:
            raise AttributeError(name)

    def as_obj(self):
        return self._config_as_obj

    def store(self, logdir):
        copy2(self._path, logdir)


class ExperimentConfig(BaseConfig):
    
    def __init__(self, path=None, from_obj=None):
        default_param_values = {
            'is_gym': False,
            'load_checkpoint': None
        }
        super().__init__(path, from_obj, default_param_values)

    def _sort_dict(self):
        self._config_as_obj["actor_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
        self._config_as_obj["critic_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
        self._config_as_obj["training"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
        self._config_as_obj['agents'].sort(key=itemgetter('algorithm_id'), reverse=False)

    def get_env_shapes(self):

        env_config = self._config_as_obj['env']
        if 'obs_size' in env_config:
            observation_shapes = [(self._config.env.obs_size,)]
            state_shapes = [(
                self._config.env.history_length,
                self._config.env.obs_size,)]
        elif 'obs_shapes' in env_config:
            observation_shapes = self._config.env.obs_shapes
            state_shapes = []
            for obs_shape in observation_shapes:
                state_shapes = [
                    tuple([self._config.env.history_length] + obs_shape)
                ]
        else:
            raise NotImplementedError()

        if 'obs_dtypes' in env_config:
            obs_dtypes = self._config.env.obs_dtypes
        else:
            obs_dtypes = [('float32',) for _ in observation_shapes]

        action_size = self._config.env.action_size
        return observation_shapes, obs_dtypes, state_shapes, action_size

    def is_ensemble(self):
        return False


class EnsembleExperimentConfig(ExperimentConfig):
    def _sort_dict(self):
        for algo in self._config_as_obj['ensemble']['algorithms']:
            algo["actor_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
            algo["critic_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
            algo["training"]["schedule"].sort(key=itemgetter('limit'), reverse=False)
        self._config_as_obj['agents'].sort(key=itemgetter('algorithm_id'), reverse=False)

    def is_ensemble(self):
        return True

    def get_algo_config(self, index):
        algo_config_obj = copy.deepcopy(self.as_obj())
        algo_config_obj.update(algo_config_obj['ensemble']['algorithms'][index].items())
        del algo_config_obj['ensemble']
        return ExperimentConfig(from_obj=algo_config_obj)


def load_config(path):
    config_obj = load_yaml(path)
    # if isinstance(config_obj, list):
    #     return RunAgentsConfig(path)
    if 'ensemble' in config_obj:
        return EnsembleExperimentConfig(path)
    else:
        return ExperimentConfig(path)
