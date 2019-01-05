import time
from datetime import datetime

from tensorboardX import SummaryWriter

from misc.common import create_if_need


class RLLogger:
    def __init__(
        self,
        logdir,
        agent_id,
        validation,
        env
    ):
        self._logdir = logdir
        self._agent_id = agent_id
        self._env = env
        self._validation = validation
        path_to_rewards_train = f"{self._logdir}/rewards-train-{self._agent_id}.txt"
        path_to_rewards_test = f"{self._logdir}/rewards-test-{self._agent_id}.txt"

        current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
        if validation:
            self._path_to_rewards = path_to_rewards_test
            logpath = f"{self._logdir}/agent-valid-{self._agent_id}-{current_date}"
        else:
            self._path_to_rewards = path_to_rewards_train
            logpath = f"{self._logdir}/agent-train-{self._agent_id}-{current_date}"
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
        self._logger.add_scalar(
            "reward", self._env.get_total_reward(), episode_index)
        self._logger.add_scalar(
            "episode per minute",
            episode_index / elapsed_time * 60,
            episode_index)
        self._logger.add_scalar(
            "steps per second",
            n_steps / elapsed_time,
            episode_index)

        with open(self._path_to_rewards, "a") as f:
            f.write(
                str(self._agent_id) + " " +
                str(episode_index) + " " +
                str(self._env.get_total_reward()) + "\n")
                
        self._start_time = time.time()


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
