# will be reworked to use with redis

import queue
from time import sleep
from multiprocessing import Process, Queue, Value

import numpy as np

from rl_server.server.server_replay_buffer import ServerBuffer, Transition
from rl_server.server.rl_client import episode_to_req
from rl_server.server.rl_server_api import req_to_episode


def np_array_to_str(arr):
    return {
        'data': arr.reshape((-1)).tostring(),
        'shape': arr.shape
    }
    
    
def str_to_np_array(data, dtype):
    return np.frombuffer(
        data['data'],
        dtype=dtype
    ).reshape(data['shape'])


def batch_to_str(batch):
    return [
        np_array_to_str(batch.s),
        np_array_to_str(batch.a),
        np_array_to_str(batch.r),
        np_array_to_str(batch.s_),
        np_array_to_str(batch.done)
    ]
    
    
def str_to_batch(batch):
    return Transition(
        str_to_np_array(batch[0], np.float32),
        str_to_np_array(batch[1], np.float32),
        str_to_np_array(batch[2], np.float32),
        str_to_np_array(batch[3], np.float32),
        str_to_np_array(batch[4], np.bool)
    )


class ParallelBatchFetcher:
    
    def __init__(self, buffer_size, observation_shapes, action_size, batch_prepare_params):
        self._buffer_size = buffer_size
        self._observation_shapes = observation_shapes
        self._action_size = action_size
        self._batch_prepare_params = batch_prepare_params

        self._stop_batch_loop = Value('i', 0)
        self._output_batch_queue = Queue(100)
        self._input_episode_queue = Queue(100)
        self._stored_in_buffer = Value('i', 0)
        
    def start(self):

        def batch_prepare_loop(
            buffer_size,
            observation_shapes,
            action_size,
            batch_prepare_params,
            stop_batch_loop,
            input_episode_queue,
            output_batch_queue,
            stored_in_buffer
        ):
            batch_size = batch_prepare_params['batch_size']
            history_len = batch_prepare_params['history_len']
            n_step = batch_prepare_params['n_step']
            gamma = batch_prepare_params['gamma']
            indices = batch_prepare_params['indices']
        
            server_buffer = ServerBuffer(
                buffer_size,
                observation_shapes,
                action_size)
                
            print('--- start')
            while stop_batch_loop.value == 0:
                # check new episodes to put
                try:
                    episode = input_episode_queue.get_nowait()
                    server_buffer.push_episode(episode)
                    stored_in_buffer.value = server_buffer.get_stored_in_buffer()
                except queue.Empty:
                    pass
                
                # fetch new batch
                if stored_in_buffer.value > 5000:
                    batch = server_buffer.get_batch(
                        batch_size,
                        history_len,
                        n_step,
                        gamma,
                        indices)
                    batch = batch_to_str(batch)
                    try:
                        output_batch_queue.put(batch, block=True, timeout=1.0)
                    except queue.Full:
                        pass
            
        self._batch_loop_process = Process(
            target=batch_prepare_loop,
            args=(
                self._buffer_size,
                self._observation_shapes,
                self._action_size,
                self._batch_prepare_params,
                self._stop_batch_loop,
                self._input_episode_queue,
                self._output_batch_queue,
                self._stored_in_buffer
            ))
            
        self._batch_loop_process.start()
        
    def join(self):
        self._batch_loop_process.join()
        
    def push_episode(self, episode):
        try:
            self._input_episode_queue.put(episode, block=True, timeout=1.0)
        except queue.Full:
            print('--- cannot store episode, episode queue is full')
        
    def get_batch(self):
        try:
            batch = self._output_batch_queue.get(block=True, timeout=1.0)
            batch = str_to_batch(batch)
            return batch
        except queue.Empty:
            return None

    def get_stored_in_buffer(self):
        return self._stored_in_buffer.value
