import numpy as np
from .tcp_client_server import TCPServer
from .serialization import serialize, deserialize


def preprocess_states(states, shapes):
    for s in states:
        for i in range(len(s)):
            s[i] = np.fromstring(s[i], dtype=np.float32).reshape(shapes[i])


class RLServerAPI(object):

    def __init__(
            self,
            num_clients,
            observation_shapes,
            observation_shapes_low_level=None,
            ip_address_str='0.0.0.0',
            first_client_port=8777,
            network_timeout=120):

        self._num_clients = num_clients
        self._observation_shapes = observation_shapes
        self._observation_shapes_low_level = observation_shapes_low_level
        self._ip_address_str = ip_address_str
        self._first_client_port = first_client_port
        self._network_timeout = network_timeout

    def set_act_batch_callback(self, callback):
        self._act_batch_callback = callback

    def set_act_test_controller_batch_callback(self, callback):
        self._act_test_controller_batch_callback = callback

    def set_store_exp_batch_callback(self, callback):
        self._store_exp_batch_callback = callback

    def start_server(self):
        for i in range(self._num_clients):
            server = TCPServer(
                self._ip_address_str,
                self._first_client_port + i,
                self._network_timeout)
            server.listen(self.agent_listener)

    def agent_listener(self, request):
        req = deserialize(request)
        method = req['method']

        if method == 'act_batch':
            preprocess_states(req['states'], self._observation_shapes)
            response = self._act_batch_callback(req['states'])

        elif method == 'act_test_controller_batch':
            assert self._observation_shapes_low_level is not None, \
                'if you want to test low level controller \
                specify observation_shapes_low_level'
            preprocess_states(
                req['states'], self._observation_shapes_low_level)
            response = self._act_test_controller_batch_callback(req['states'])

        elif method == 'store_exp_batch':
            preprocess_states(req['prev_states'], self._observation_shapes)
            preprocess_states(req['next_states'], self._observation_shapes)
            self._store_exp_batch_callback(
                req['rewards'],
                req['actions'],
                req['prev_states'],
                req['next_states'],
                req['is_terminators'])
            response = ''

        return serialize(response)
