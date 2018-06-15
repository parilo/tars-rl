import threading
from .tcp_client_server import TCPClient
from .serialization import serialize, deserialize


class RLClient(object):

    def __init__(self, port=8777):
        self._tcp_client = TCPClient('127.0.0.1', port, 120)
        self._tcp_client.connect()
        self._tcp_lock = threading.Lock()

    def preprocess_states(self, states):
        for s in states:
            for i in range(len(s)):
                s[i] = s[i].reshape((-1)).tostring()

    def copy_states(self, states):
        states_copy = list(states)
        for i in range(len(states)):
            states_copy[i] = list(states[i])
        return states_copy

    def act(self, state):
        """
            state is list of state parts
            in case you have many modalities in
            your state and want to process it
            differently in the NN
        """
        return self.act_batch([state])[0]

    def act_batch(self, states):
        """
            state is list of state parts
            in case you have many modalities in
            your state and want to process it
            differently in the NN
        """
        states = self.copy_states(states)
        self.preprocess_states(states)

        req = serialize({
            'method': 'act_batch',
            'states': states
        })

        with self._tcp_lock:
            data = self._tcp_client.write_and_read_with_retries(req)
            return deserialize(data)

    def act_test_controller_batch(self, states):
        states = self.copy_states(states)
        self.preprocess_states(states)

        req = serialize({
            'method': 'act_test_controller_batch',
            'states': states
        })

        with self._tcp_lock:
            data = self._tcp_client.write_and_read_with_retries(req)
            return deserialize(data)

    def store_exp(
            self,
            reward,
            action,
            prev_state,
            next_state,
            is_terminator):

        assert isinstance(action, list), 'action must be list'
        assert isinstance(prev_state, list), 'prev_state must be list'
        assert isinstance(next_state, list), 'next_state must be list'

        self.store_exp_batch(
            [reward],
            [action],
            [prev_state],
            [next_state],
            [is_terminator]
        )

    def store_exp_batch(
            self,
            rewards,
            actions,
            prev_states,
            next_states,
            is_terminators):

        assert isinstance(actions, list), 'actions must be list'
        assert isinstance(prev_states, list), 'prev_state must be list'
        assert isinstance(next_states, list), 'next_state must be list'

        prev_states = self.copy_states(prev_states)
        next_states = self.copy_states(next_states)
        self.preprocess_states(prev_states)
        self.preprocess_states(next_states)

        req = serialize({
            'method': 'store_exp_batch',
            'rewards': rewards,
            'actions': actions,
            'prev_states': prev_states,
            'next_states': next_states,
            'is_terminators': is_terminators
        })

        with self._tcp_lock:
            self._tcp_client.write_and_read_with_retries(req)
