import os
import numpy as np
import tempfile
import tensorflow as tf
import time
from .experience_replay_buffer import ExperienceReplayBuffer
from replay_buffer import ServerBuffer
from threading import Lock

class RLTrainLoop():

    def __init__(self, observation_shapes, buffer_size=1000000):
        
        self._observation_shapes = observation_shapes
        self._buffer_size = buffer_size
        self.server_buffer = ServerBuffer(capacity=self._buffer_size,
                                          history_len=3,
                                          num_of_parts_in_obs = len(observation_shapes))
        self._store_lock = Lock()
        self._store_index = 0

    def get_observation_shapes(self):
        return self._observation_shapes
    
    def store_exp_episode(self, episode):
        
        with self._store_lock:
            
            print ("Episode received")
            num_obs = len(episode[1])
            print ("Number of observations:", num_obs)
            i = np.random.randint(num_obs)
            print ("Transition number", str(i)+":")
            for part_id in range(len(episode[0])):
                print ("Observation part", str(part_id)+":", episode[0][part_id][i])
            print ("Action:", episode[1][i])
            print ("Reward:", episode[2][i])
            print ("Done:", episode[3][i])
            
            self.server_buffer.push_episode(episode)
            
            batch = self.server_buffer.get_batch(3)
            
            for part_id in range(len(batch.s)):
                print ("State part", str(part_id)+":", batch.s[part_id])
            print ("Action:", batch.a)
            print ("Reward:", batch.r)
            for part_id in range(len(batch.s)):
                print ("Next state part", str(part_id)+":", batch.s_[part_id])
            print ("Done:", batch.done)