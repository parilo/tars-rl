import threading

import numpy as np

from .tcp_client_server import TCPClient
from .serialization import serialize, deserialize


def obs_to_string(observations):
    """ Convert observations (or states) to strings for transmission to server.

    Parameters
    ----------
    observations: list of np.arrays [obs_1, ..., obs_n]
        which corresponds to the original observations (or states)
        each np.array obs_i has shape (batch_size) + obs_shape_i

    Returns
    -------
    str_obs: list of strings [str_1, ..., str_n]
        which corresponds to the encoded observations (or states)
    """
    str_obs = []
    for obs in observations:
        str_obs.append(obs.reshape(-1).tostring())
    return str_obs


def episode_to_req(episode, method="store_episode"):
    """ Create compact serialized representation of the episode
        to pass it as a request.
    """
    observations, actions, rewards, dones = episode
    str_obs = obs_to_string(observations)
    str_act = actions.tolist()
    str_rew = rewards.tolist()
    str_don = dones.tolist()
    req = {"method": method,
                     "observations": str_obs,
                     "actions": str_act,
                     "rewards": str_rew,
                     "dones": str_don}
    return req
    

def string_to_weights(weights):
    for nn_name in weights:
        nn_weights = weights[nn_name]
        for i in range(len(nn_weights)):
            data = nn_weights[i]
            nn_weights[i] = np.frombuffer(
                data['data'],
                dtype=np.float32
            ).reshape(data['shape'])


class RLClient:

    def __init__(self, ip_address="127.0.0.1", port=8777, network_timeout=120):
        """ Class for RL Client which interacts with RL Server.

        Parameters
        ----------
        ip_address: str
            ip address of the client
        port: int
            port number of the client
        network_timeout: int
            network timeout
        """
        self._tcp_client = TCPClient(ip_address, port, network_timeout)
        self._tcp_client.connect()
        self._tcp_lock = threading.Lock()

    def store_episode(self, episode):
        req = serialize(episode_to_req(episode, method="store_episode"))
        with self._tcp_lock:
            self._tcp_client.write_and_read_with_retries(req)

    def get_weights(self, index=0):
        req = serialize({"method": "get_weights", "index": index})
        with self._tcp_lock:
            data = self._tcp_client.write_and_read_with_retries(req)
            data = deserialize(data)
            string_to_weights(data)
            return data
