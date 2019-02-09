import time
from threading import Lock

from rl_server.server.server_replay_buffer import ServerBuffer
from misc.rl_logger import RLServerLogger


class RLTrainer:
    def __init__(
            self,
            observation_shapes,
            observation_dtypes,
            action_size,
            experience_replay_buffer_size=1000000,
            use_prioritized_buffer=False,
            n_step=1,
            gamma=0.99,
            train_every_nth=4,
            history_length=3,
            initial_beta=0.4,
            target_critic_update_period=500,
            target_actor_update_period=500,
            start_learning_after=5000,
            show_stats_period=2000,
            save_model_period=10000,
            logdir="ckpt/"):

        self._observation_shapes = observation_shapes
        self._observation_dtypes = observation_dtypes
        self._action_size = action_size
        self._buffer_size = experience_replay_buffer_size
        self._start_learning_after = start_learning_after
        self._n_step = n_step
        self._gamma = gamma
        self._train_every = train_every_nth
        self._target_critic_update_period = target_critic_update_period
        self._target_actor_update_period = target_actor_update_period
        self._show_stats_period = show_stats_period
        self._save_model_period = save_model_period
        self._hist_len = history_length
        self._beta = initial_beta
        self._use_prioritized_buffer = use_prioritized_buffer
        self._logdir = logdir
        self._logger = RLServerLogger(logdir)

        # sync buffer
        self.server_buffer = ServerBuffer(
            self._buffer_size,
            observation_shapes,
            observation_dtypes,
            action_size
        )
        self._train_loop_step_lock = Lock()

        self._step_index = 0
        self._target_actor_update_num = 0
        self._target_critic_update_num = 0
        self._n_saved = 0

    def set_algorithm(self, algo):
        self._algo = algo

    def init(self):
        pass

    def store_episode(self, episode):
        self.server_buffer.push_episode(episode)

    # for asynchronous acts and trains
    def start_training(self):
        while True:
            buffer_size = self.server_buffer.get_stored_in_buffer()
            if buffer_size > self._start_learning_after:
                if buffer_size > self._step_index * self._train_every:
                    self.train_step()
                    self._step_index += 1
            elif buffer_size < self._start_learning_after \
                    and self._step_index % 10 == 0:
                print("--- buffer size {}".format(buffer_size))
                time.sleep(1.0)

    def train_step(self):
        pass

    def get_weights(self):
        return None
