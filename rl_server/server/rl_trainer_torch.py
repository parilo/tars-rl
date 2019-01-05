import os
import torch
import time
from rl_server.server.server_replay_buffer import ServerBuffer
from threading import Lock, Thread
import multiprocessing
from tensorboardX import SummaryWriter
from datetime import datetime
from rl_server.server.rl_trainer import RLTrainer


class TorchRLTrainer(RLTrainer):
    def save(self):
        if self._step_index % self._save_model_period == 0:
            actor_state_dict = self._algo._actor.state_dict()
            filename = "{logdir}/actor.{suffix}.pth.tar".format(
                logdir=self._logdir, suffix=str(self._step_index))
            torch.save(actor_state_dict, filename)
            print("Actor saved in: %s" % filename)

            criitc_state_dict = self._algo._critic.state_dict()
            filename = "{logdir}/critic.{suffix}.pth.tar".format(
                logdir=self._logdir, suffix=str(self._step_index))
            torch.save(criitc_state_dict, filename)
            print("Critic saved in: %s" % filename)

            self._n_saved += 1
            self._logger.add_scalar(
                "num saved",
                self._n_saved,
                self._step_index)
