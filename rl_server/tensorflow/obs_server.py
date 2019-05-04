import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from rl_server.server.rl_server_api import RLServerAPI


class ObsServer:

    def __init__(self, exp_config):
        (
            observation_shapes,
            observation_dtypes,
            state_shapes,
            action_size
        ) = exp_config.get_env_shapes()

        self._server_api = RLServerAPI(
            exp_config.obs_server.num_clients,
            observation_shapes,
            observation_dtypes,
            state_shapes,
            init_port=exp_config.obs_server.client_start_port)

        self._server_api.set_preprocess_obs_callback(self._preprocess_obs)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(sess)
        self._vae_enc = load_model(exp_config.obs_server.encoder_model_path)
        self._graph = tf.get_default_graph()

    def start(self):
        print("--- starting obs server")
        self._server_api.start_server()

    def _preprocess_obs(self, obs_list):
        #print('--- preprocess', obs_list[0].shape)
        with self._graph.as_default():
            output = []
            for obs in obs_list:
                obs = np.transpose(obs, (2, 0, 1)) / 255.
                output.append(self._vae_enc.predict(x=np.expand_dims(obs, axis=0), batch_size=1)[0])
            return output
