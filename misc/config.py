from operator import itemgetter

import yaml


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.load(f.read())

    
def load_ensemble_config(path):
    config = load_yaml(path)
    algo_configs = []
    for algo_config_path in config['ensemble']['algo_config_paths']:
        algo_configs.append(load_yaml(algo_config_path))
    return config, algo_configs


class EnsembleConfig:
    
    def __init__(self, path):
        self.config, self.algo_configs = load_ensemble_config(path)
        for algo_config in self.algo_configs:
            algo_config["actor_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=True)
            algo_config["critic_optim"]["schedule"].sort(key=itemgetter('limit'), reverse=True)
            algo_config["training"]["schedule"].sort(key=itemgetter('limit'), reverse=True)
    
    def get_env_shapes(self):
        observation_shapes = [(self.config["env"]["obs_size"],)]
        state_shapes = [(
            self.config["env"]["history_length"],
            self.config["env"]["obs_size"],)]
        action_size = self.config["env"]["action_size"]
        return observation_shapes, state_shapes, action_size
        
    def get_algos_count(self):
        return len(self.algo_configs)
