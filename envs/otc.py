import random

import numpy as np
import cv2
from obstacle_tower_env import ObstacleTowerEnv
import tensorflow as tf
from tensorflow.keras.models import load_model

from envs.gym_env import GymEnvWrapper
from rl_server.server.rl_client import RLClient


class ObstacleTowerEnvWrapper(GymEnvWrapper):

    def __init__(
        self,
        environment_filename,
        retro,
        agent_id,
        start_level_inc=None,
        grayscale=False,
        vae_path=None,
        obs_server=None,
        **kwargs
    ):
        self._env = ObstacleTowerEnv(
            environment_filename,
            retro=retro,
            worker_id=agent_id
        )

        self._retro = retro
        self._current_level = 0
        self._start_level_inc = start_level_inc
        self._grayscale = grayscale
        self._vae_enc = None
        self._obs_client = None

        self._key_ind = [
    3,    8,   13,   16,   23,   31,   33,   34,   35,   38,   42,   47,   50,   54,
   56,   65,   80,   81,   85,   89,  105,  112,  120,  126,  127,  134,  135,  137,
  143,  145,  148,  157,  163,  164,  165,  172,  180,  185,  187,  192,  193,  200,
  201,  202,  206,  208,  213,  214,  217,  229,  232,  240,  243,  249,  255,  258,
  259,  264,  271,  276,  277,  284,  299,  301,  302,  305,  307,  315,  320,  327,
  328,  336,  339,  342,  346,  350,  354,  355,  362,  372,  379,  384,  386,  390,
  394,  405,  409,  412,  414,  415,  420,  423,  430,  436,  437,  440,  442,  445,
  446,  448,  450,  455,  456,  458,  461,  464,  478,  479,  484,  501,  504,  509,
  511,  513,  514,  515,  526,  529,  535,  546,  548,  557,  558,  559,  563,  565,
  567,  575,  577,  578,  583,  588,  589,  590,  604,  605,  608,  614,  618,  619,
  637,  639,  645,  651,  657,  658,  659,  663,  665,  669,  674,  679,  680,  681,
  687,  693,  695,  696,  697,  699,  700,  711,  717,  719,  732,  739,  746,  757,
  761,  762,  765,  769,  770,  774,  780,  781,  785,  786,  787,  788,  792,  794,
  814,  815,  820,  826,  827,  835,  838,  848,  859,  870,  875,  879,  882,  888,
  889,  890,  893,  901,  902,  905,  906,  910,  913,  916,  922,  933,  938,  942,
  944,  949,  951,  963,  966,  972,  973,  974,  998, 1001, 1002, 1005, 1007, 1009,
 1010, 1013, 1017, 1018, 1023
]

        self._action_remap_reverse = {
            0: 0,  # no
            3: 1,  # j
            6: 2,  # cam left
            12: 3,  # cam right
            18: 4,  # forward
            21: 5,  # jump + forward
            36: 6  # backward
        }

        if vae_path:
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            tf.keras.backend.set_session(sess)
            self._vae_enc = load_model(vae_path)
            self._keras_graph = tf.get_default_graph()
        if obs_server:
            obs_server['port'] += agent_id
            self._obs_client = RLClient(**obs_server)

        super().__init__(env=self._env, agent_id=agent_id, **kwargs)

    def reset(self):
        if self._start_level_inc is None:
            self._current_level = 0
        else:
            self._current_level = min(max(0, self._current_level + self._start_level_inc), 24)
        self.env.floor(self._current_level)
        #floor = random.randint(0, 24)
        #print('--- floor', floor)
        #self.env.floor(floor)
        return self.process_obs(super().reset())

    # def split_observation(self, obs):
    #     obs_parts = []
    #     for i in range(3):
    #         for j in range(3):
    #             obs_parts.append(obs[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28])
    #     return np.stack(obs_parts, axis=0)
    #
    # def preprocess_obs(self, obs):
    #     obs = super().preprocess_obs(obs)
    #     return self.split_observation(obs)

    def process_obs(self, observation, info=None, reward=None, action=None):
        if info:
            vec_obs = np.copy(info['brain_info'].vector_observations[0])
            vec_obs[6] /= 3000.
            vec_obs = np.array(vec_obs.tolist() + [reward], dtype=np.float32)
        else:
            vec_obs = np.zeros((8,), dtype=np.float32)
            vec_obs[0] = 1
            vec_obs[6] = 1
        # if self._vae_enc:
        #     with self._keras_graph.as_default():
        #         obs = np.transpose(observation, (2, 0, 1)) / 255.
        #         return [
        #             self._vae_enc.predict(x=np.expand_dims(obs, axis=0), batch_size=1)[0],
        #             # vec_obs
        #         ]
        # if self._obs_client:
        #     obs = self._obs_client.preprocess_obs([
        #         cv2.resize(observation, (168, 168))
        #     ])[0][0, self._key_ind]
        #     act = np.zeros((len(self._action_remap_reverse,)), dtype=np.uint8)
        #     # if action is not None:
        #     #     act[self._action_remap_reverse[action]] = 1
        #     return [
        #         obs,
        #         # vec_obs
        #         # act
        #     ]
        # elif self._grayscale:
        #     gs_img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        #     return [
        #         np.expand_dims(gs_img, axis=0),
        #         np.expand_dims(cv2.resize(gs_img, (32, 32)), axis=0),
        #         # vec_obs
        #     ]
        # else:
        #     # print(len(observation), observation[0].shape, observation[1], observation[2])
        return [
            np.transpose(cv2.resize(observation, (84, 84)), (2, 0, 1)),
            np.transpose(cv2.resize(observation, (32, 32)), (2, 0, 1)),
            vec_obs
        ]

    def render(self, observation):
        cv2.imshow(
            str(self.agent_id) + ' ' + str(observation[0].shape),
            # observation
            cv2.resize(observation, (400, 400))
            #cv2.resize(np.transpose(obs[0], (1, 2, 0)), (400, 400))
        )
        # cv2.imshow(
        #     str(self.agent_id) + ' ' + str(obs[1].shape),
        #     cv2.resize(np.transpose(obs[1], (1, 2, 0)), (400, 400))
        # )
        cv2.waitKey(3)

    def process_reward(self, reward):
        if reward == 1.:
            self._current_level += 1

        if reward > 0.05:
            return self.reward_scale
        else:
            return reward

    def step(self, action):
        observation, reward, done, info = super().step(action)

        obs = self.process_obs(observation, info, reward, action)
        if self.visualize:
            self.render(observation)

        # print(info['brain_info'].vector_observations)
        # if done:
        #     print('ep', self.time_step, self.total_reward, info['brain_info'].vector_observations)

        time_left = info['brain_info'].vector_observations[0][6] / 3000.
        if time_left < 0.003:  # time left
            done = True

        return obs, reward, done, info

    def get_logs(self):
        logs = super().get_logs()
        logs.update({
            'level': self._current_level
        })
        return logs











#import os
#
#import cv2
#import numpy as np
#
#
#from tensorflow.keras.models import load_model
#vae_mean = load_model('log_1/models/vae_mean_376.h5')
#
## TEST_DIR_IN = '/mnt/data/otc/vae/eps_test_train/in'
## TEST_DIR_OUT = '/mnt/data/otc/vae/eps_test_train/predict'
#TEST_DIR_IN = '/mnt/data/otc/vae/eps_images_test/in'
#TEST_DIR_OUT = '/mnt/data/otc/vae/eps_images_test/predict'
#
#
#def predict(sample):
#    pred_mean = vae_mean.predict(x=np.expand_dims(sample, axis=0), batch_size=1)[0]
#    return pred_mean
#
#
#def read_image(path):
#    return cv2.imread(path)
#
#
#def to_channel_first(sample):
#    return np.transpose(sample, (2, 0, 1))
#
#
#def to_channel_last(sample):
#    return cv2.resize(
#        np.clip(
#            np.transpose(sample, (1, 2, 0)) * 255.,
#            0,
#            255
#        ).astype(np.uint8),
#        (600, 600)
#    )
#
#
#for root, dirs, files in os.walk(TEST_DIR_IN):
#    for file in files:
#        fpath = os.path.join(TEST_DIR_IN, file)
#        print('--- process', fpath)
#
#        image = read_image(fpath)
#        pred = to_channel_last(predict(to_channel_first(image)))
#
#        images = np.concatenate(
#            [
#                cv2.resize(image, (600, 600)),
#                pred
#            ],
#            axis=1
#        )
#        cv2.imwrite(os.path.join(TEST_DIR_OUT, file), images)
#
#        # cv2.imshow('im', images)
#        # cv2.waitKey(0)
#        # break
