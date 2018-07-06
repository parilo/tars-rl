import numpy as np
from .tcp_client_server import TCPServer
from .serialization import serialize, deserialize


def string_to_obs(strings, obs_shapes):
    """ Convert strings back to observations (or states).

    Parameters
    ----------
    strings: list of strings [str_1, ..., str_n]
        which corresponds to the encoded observations (or states)
    obs_shapes: list of tuples [obs_shape_1, ..., obs_shape_n]
        shapes of corresponding observations (or states)

    Returns
    -------
    obs_str: list of np.arrays [obs_1, ..., obs_n]
        which corresponds to the original observations (or states)
        each np.array obs_i has shape (batch_size) + obs_shape_i
    """
    obs_str = []
    for i, str_ in enumerate(strings):
        obs = np.frombuffer(str_, dtype=np.float32)
        obs_str.append(obs.reshape((-1,)+obs_shapes[i]))
    return obs_str


def req_to_episode(req, obs_shapes):
    """ Preprocess deserealized request to obtain episode.
    """
    observations = string_to_obs(req['observations'], obs_shapes)
    actions = np.array(req['actions'], dtype=np.float32)
    rewards = np.array(req['rewards'], dtype=np.float32)
    dones = np.array(req['dones'], dtype=np.bool)
    return [observations, actions, rewards, dones]


class RLServerAPI:

    def __init__(self, num_clients, observation_shapes, state_shapes,
                 ip_address='0.0.0.0', init_port=8777, network_timeout=120):
        """ Class for RL Server which interacts with multiple RL Clients.

        Parameters
        ----------
        num_clients: int
            maximal number of clients
        observation_shapes: list of tuples [obs_shape_1, ..., obs_shape_n]
            which corresponds to observations' shapes
        state_shapes: list of tuples [state_shape_1, ..., state_shape_n]
            which corresponds to states' shapes
        ip_address: str
            ip address of the server
        init_port: int
            port number of the first client, all clients are assumed to be
            connected via consecutive port numbers
            [first_client_port, first_client_port+1, ...]
        network_timeout: int
            network timeout
        """

        self._num_clients = num_clients
        self._observation_shapes = observation_shapes
        self._state_shapes = state_shapes
        self._ip_address = ip_address
        self._init_port = init_port
        self._timeout = network_timeout

    def set_act_batch_callback(self, callback):
        self._act_batch_callback = callback

    def set_act_with_gradient_batch_callback(self, callback):
        self._act_with_gradient_batch_callback = callback

    def set_store_episode_callback(self, callback):
        self._store_episode_callback = callback

    def start_server(self):
        for i in range(self._num_clients):
            server = TCPServer(self._ip_address,
                               self._init_port+i,
                               self._timeout)
            server.listen(self.agent_listener)

    def agent_listener(self, request):

        req = deserialize(request)
        method = req['method']

        if method == 'act_batch':
            states = string_to_obs(req['states'], self._state_shapes)
            mode = req['mode']
            response = self._act_batch_callback(states, mode)

        elif method == 'act_with_gradient_batch':
            states = string_to_obs(req['states'], self._state_shapes)
            response = self._act_with_gradient_batch_callback(states)

        elif method == 'store_episode':
            episode = req_to_episode(req, self._observation_shapes)
            self._store_episode_callback(episode)
            response = ''

        return serialize(response)
