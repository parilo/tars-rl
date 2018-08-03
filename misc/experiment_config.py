import os
import pickle
import yaml


class ExperimentConfig:

    def __init__(
            self, env_name="prosthetics_new",
            config_path="config.yml",
            experiment_name="experiment"):

        self.env_name = env_name
        config = yaml.load(open(config_path))

        # load experiment config

        for key, value in config.items():
            setattr(self, key, value)

        self.obs_size = config["obs_size"]
        self.action_size = config["action_size"]
        self.frame_skip = config["frame_skip"]

        self.history_len = config["history_len"]
        self.n_step = config["n_step"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

        self.use_prioritized_buffer = config["use_prioritized_buffer"]

        self.use_synchronous_update = config["use_synchronous_update"]
        self.port = config["port"]

        # create file with proper experiment name
        if experiment_name == "experiment":
            experiment_file = ""
            for i in ["history_len", "frame_skip", "n_step", "batch_size"]:
                experiment_file = experiment_file + "-" + i + str(config[i])
            if self.use_prioritized_buffer:
                experiment_file = experiment_file + "-prio"
            if self.use_synchronous_update:
                experiment_file = experiment_file + "-sync"
            else:
                experiment_file = experiment_file + "-async"
            self.path_to_experiment = "results/" + experiment_file + "/"
        else:
            self.path_to_experiment = "results/" + experiment_name + "/"

        self.path_to_ckpt = self.path_to_experiment + "ckpt/"
        self.path_to_rewards_train = self.path_to_experiment + "rewards-train.txt"
        self.path_to_rewards_test = self.path_to_experiment + "rewards-test.txt"

    def save_info(self, info):
        if not os.path.exists(self.path_to_experiment):
            os.makedirs(self.path_to_experiment)
        with open(self.path_to_experiment + "info.pkl", "wb") as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

    def load_info(self):
        with open(self.path_to_experiment + "info.pkl", "rb") as f:
            return pickle.load(f)
