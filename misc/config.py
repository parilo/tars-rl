from operator import itemgetter

import yaml


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.load(f.read())


class ExperimentConfig:
    
    def __init__(self, path):
        self._config = load_yaml(path)
        if self.is_ensamble():
            self._algo_configs = [
                load_yaml(path)
                for path in self._config['ensemble']['algo_config_paths']
            ]
        else:
            self._algo_configs = [
                load_yaml(self._config['algo_config_path'])
            ]
        # self.config, self.algo_configs = load_ensemble_config(path)
        for algo_config in self._algo_configs:
            algo_config["actor_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=True)
            algo_config["critic_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=True)
            algo_config["training"]["schedule"].sort(key=itemgetter('limit'), reverse=True)

    def is_ensamble(self):
        return hasattr(self._config, 'ensemble')
    
    def get_env_shapes(self):
        observation_shapes = [(self._config["env"]["obs_size"],)]
        state_shapes = [(
            self._config["env"]["history_length"],
            self._config["env"]["obs_size"],)]
        action_size = self._config["env"]["action_size"]
        return observation_shapes, state_shapes, action_size
        
    def get_algos_count(self):
        return len(self._algo_configs)

    def get_algo_config(self, index):
        return self._algo_configs[index]
