import os
import pickle
import json


class ExperimentConfig:

    def __init__(self, env_name='prosthetics_new', 
                 config_path='config.txt',
                 experiment_name='experiment'):

        self.env_name = env_name
        config = json.load(open(config_path))

        # load experiment config    
        self.history_len = config['history_len']
        self.frame_skip = config['frame_skip']
        self.n_step = config['n_step']
        self.obs_size = config['observation_size']
        self.action_size = config['action_size']
        self.use_prioritized_buffer = config['prio']
        self.batch_size = config['batch_size']
        self.prio = config['prio']
        self.use_synchronous_update = config['sync']
        self.port = config['port']
        self.gpu_id = config['gpu_id']
        self.disc_factor = config['disc_factor']

        # create file with proper experiment name
        if experiment_name == 'experiment':
            experiment_file = ''
            for i in ['history_len', 'frame_skip', 'n_step', 'batch_size']:
                experiment_file = experiment_file + '-' + i + str(config[i])
            if self.prio:
                experiment_file = experiment_file + '-prio'
            if self.use_synchronous_update:
                experiment_file = experiment_file + '-sync'
            else:
                experiment_file = experiment_file + '-async'
            self.path_to_experiment = 'results/' + experiment_file + '/'
        else:
            self.path_to_experiment = 'results/' + experiment_name + '/'

        self.path_to_ckpt = self.path_to_experiment + 'ckpt/'
        self.path_to_rewards_train = self.path_to_experiment + 'rewards-train.txt'
        self.path_to_rewards_test = self.path_to_experiment + 'rewards-test.txt'

    def save_info(self, info):
        if not os.path.exists(self.path_to_experiment):
            os.mkdir(self.path_to_experiment)
        with open(self.path_to_experiment + 'info.pkl', 'wb') as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

    def load_info(self):
        with open(self.path_to_experiment + 'info.pkl', 'rb') as f:
            return pickle.load(f)
