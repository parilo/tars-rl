import tensorflow as tf

from rl_server.server.rl_trainer_tf import make_session
from rl_server.tensorflow.algo.algo_fabric import create_algorithm


class AgentModel:
    
    def __init__(self, experiment_config, agent_config, rl_client):

        observation_shapes, state_shapes, action_size = experiment_config.get_env_shapes()
        self._rl_client = rl_client
        self._agent_algorithm = create_algorithm(
            observation_shapes,
            state_shapes,
            action_size,
            agent_config
        )
        self._sess = make_session(num_cpu=1)
        
    def set_weights(self, weights):
        self._agent_algorithm.set_weights(self._sess, weights)
        
    def act_batch(self, states, mode="default"):
        if mode == "default":
            actions = self._agent_algorithm.act_batch(self._sess, states)
        elif mode == "sac_deterministic":
            actions = self._agent_algorithm.act_batch_deterministic(self._sess, states)
        elif mode == "with_gradients":
            # actually, here it will return actions and grads
            actions = self._agent_algorithm.act_batch_with_gradients(self._sess, states)
        else:
            raise NotImplementedError
        return actions
        
    def fetch(self, index=0):
        self.set_weights(self._rl_client.get_weights(index=index))
