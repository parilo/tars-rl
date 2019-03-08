import time
from datetime import datetime

from tensorboardX import SummaryWriter
import numpy as np

from misc.common import create_if_need


class RLLogger:
    def __init__(
        self,
        logdir,
        agent_id,
        validation,
        env,
        log_every_n_steps
    ):
        self._logdir = logdir
        self._agent_id = agent_id
        self._env = env
        self._validation = validation
        self._log_every_n_steps = log_every_n_steps
        self._steps_after_last_logging = 0
        self._ep_rewards_after_last_logging = []
        self._ep_logged_count = 0
        path_to_rewards_train = "{}/rewards-train-{}.txt".format(self._logdir, self._agent_id)
        path_to_rewards_test = "{}/rewards-test-{}.txt".format(self._logdir, self._agent_id)

        current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
        if validation:
            self._path_to_rewards = path_to_rewards_test
            logpath = "{}/agent-valid-{}-{}".format(self._logdir, self._agent_id, current_date)
        else:
            self._path_to_rewards = path_to_rewards_train
            logpath = "{}/agent-train-{}-{}".format(self._logdir, self._agent_id, current_date)
        create_if_need(logpath)
        self._logger = SummaryWriter(logpath)
        self._start_time = time.time()
        
    def log(self, episode_index, n_steps):
        elapsed_time = time.time() - self._start_time
        if self._validation:
            print("--- val episode ended {} {} {} {} {}".format(
                self._agent_id, episode_index, self._env.time_step,
                self._env.get_total_reward(), self._env.get_total_reward_shaped()))

        self._logger.add_scalar("steps", n_steps, episode_index)

        # for fair comparison we
        # need to log episode reward per steps
        # so for now logging one time per 1000 steps
        self._steps_after_last_logging += n_steps
        self._ep_rewards_after_last_logging.append(self._env.get_total_reward())
        while self._steps_after_last_logging >= self._log_every_n_steps:
            self._steps_after_last_logging -= self._log_every_n_steps
            mean_ep_reward = np.mean(self._ep_rewards_after_last_logging)
            self._logger.add_scalar(
                "reward", mean_ep_reward, self._ep_logged_count)

            with open(self._path_to_rewards, "a") as f:
                f.write(
                    str(self._agent_id) + " " +
                    str(self._ep_logged_count) + " " +
                    str(mean_ep_reward) + "\n")
            self._ep_logged_count += 1
        self._ep_rewards_after_last_logging = []

        self._logger.add_scalar(
            "episode per minute",
            episode_index / elapsed_time * 60,
            episode_index)
        self._logger.add_scalar(
            "steps per second",
            n_steps / elapsed_time,
            episode_index)

        self._start_time = time.time()

    def log_dict(self, log_dict, episode_index):
        for key, val in log_dict.items():
            self._logger.add_scalar(key, val, episode_index)


class RLServerLogger:
    def __init__(
        self,
        logdir
    ):
        self._logdir = logdir
        current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
        logpath = "{}/server-{}".format(logdir, current_date)
        create_if_need(logpath)
        self._logger = SummaryWriter(logpath)

    def log_buffer_size(self, buffer_size, step_index):
        self._logger.add_scalar('buffer_size', buffer_size, step_index)

    def log_train(self, train_info, step_index):
        if isinstance(train_info, dict):
            for key, value in train_info.items():
                self._logger.add_scalar('train_info_' + key, value, step_index)
        elif isinstance(train_info, list):
            for i, value in enumerate(train_info):
                self._logger.add_scalar('train_info_' + str(i), value, step_index)
        else:
            self._logger.add_scalar('train_info', train_info, step_index)
