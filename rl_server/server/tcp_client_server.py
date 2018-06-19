# Copyright (c) 2017 Computer Vision Center (CVC)
# at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic TCP client and server"""

import logging
import socket
import struct
import time
import threading
from io import BytesIO


class TCPConnectionError(Exception):
    pass


class TCPConnectionClosedError(TCPConnectionError):
    pass


class TCPBase(object):
    """
    Basic networking client for TCP connections. Errors occurred during
    networking operations are raised as TCPConnectionError.

    Received messages are expected to be prepended by a int32 defining the
    message size. Messages are sent following this convention.
    """

    def __init__(self, host, port, timeout):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket = None
        self._logprefix = '--- (%s:%s) ' % (self._host, self._port)

    def disconnect(self):
        """Disconnect any active connection."""
        if self._socket is not None:
            logging.debug(self._logprefix + 'disconnecting')
            self._socket.close()
            self._socket = None

    def connected(self):
        """Return whether there is an active connection."""
        return self._socket is not None

    def write(self, message):
        """Send message to the server."""
        if self._socket is None:
            raise TCPConnectionError(self._logprefix + 'not connected')
        header = struct.pack('<L', len(message))
        try:
            self._socket.sendall(header + message)
        except Exception as exception:
            print('--- tcp write: error {}'.format(exception))
            self._reraise_exception_as_tcp_error(
                'failed to write data', exception)

    def read(self):
        """Read a message from the server."""
        header = self._read_n(4)
        if not header:
            raise TCPConnectionClosedError(self._logprefix + 'connection closed')
        length = struct.unpack('<L', header)[0]
        data = self._read_n(length)
        return data

    def _read_n(self, length):
        """Read n bytes from the socket."""
        if self._socket is None:
            raise TCPConnectionError(self._logprefix + 'not connected')
        buf = BytesIO()
        while length > 0:
            try:
                data = self._socket.recv(length)
            except Exception as exception:
                self._reraise_exception_as_tcp_error(
                    'failed to read data', exception)
            if not data:
                raise TCPConnectionClosedError(self._logprefix + 'connection closed')
            buf.write(data)
            length -= len(data)
        return buf.getvalue()

    def _reraise_exception_as_tcp_error(self, message, exception):
        raise TCPConnectionError(
            '%s%s: %s' % (self._logprefix, message, exception))

    def write_and_read_with_retries(self, data):
        while True:
            try:
                self.write(data)
                data = self.read()
                return data
            except TCPConnectionClosedError as e:
                raise e
            except TCPConnectionError as e:
                print('--- write and read tcp error {} retring'.format(e))


class TCPClient(TCPBase):
    def connect(self, connection_attempts=10):
        """Try to establish a connection to the given host:port."""
        connection_attempts = max(1, connection_attempts)
        error = None
        for attempt in range(1, connection_attempts + 1):
            try:
                self._socket = socket.create_connection(
                    address=(self._host, self._port),
                    timeout=self._timeout)
                self._socket.settimeout(self._timeout)
                logging.debug(self._logprefix + 'connected')
                return
            except Exception as exception:
                error = exception
                logging.debug(
                    self._logprefix + 'connection attempt %d: %s',
                    attempt,
                    error)
                time.sleep(1)
                continue
        self._reraise_exception_as_tcp_error('failed to connect', error)


class TCPServer(TCPBase):
    def listen(self, callback):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self._host, self._port))

        def communication_func():
            while True:
                try:
                    request = self.read()
                    response = callback(request)
                    self.write(response)
                except TCPConnectionClosedError as e:
                    print('--- tcp connection closed error {}'.format(e))
                    break
                except TCPConnectionError as e:
                    print('--- tcp server: tcp connection error {} retring to read'.format(e))

        def listen_func():
            while True:
                server_socket.listen()
                self._socket, addr = server_socket.accept()
                self._socket.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._socket.settimeout(self._timeout)
                communication_func()

        listen_thread = threading.Thread(target=listen_func)
        listen_thread.start()
