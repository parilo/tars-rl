
from rl_server.algo.algo_fabric import create_algorithm


class AgentModel:
    
    def __init__(self, exp_config, rl_client):
        self._rl_client = rl_client
        self._agent_algorithm = create_algorithm(exp_config)

    def set_weights(self, weights):
        self._agent_algorithm.set_weights(weights)
        
    def act_batch(self, states, mode="default"):
        if mode == "default":
            actions = self._agent_algorithm.act_batch(states)
        elif mode == "deterministic":
            actions = self._agent_algorithm.act_batch_deterministic(states)
        elif mode == "with_gradients":
            # actually, here it will return actions and grads
            actions = self._agent_algorithm.act_batch_with_gradients(states)
        else:
            raise NotImplementedError
        return actions

    def act_batch_target(self, states):
        return self._agent_algorithm.act_batch_target(states)

    def fetch(self, algo_index=0):
        self.set_weights(self._rl_client.get_weights(algo_index=algo_index))

    def load_checkpoint(self, load_info):
        self._agent_algorithm.load(load_info.dir, load_info.index)

    def reset_states(self):
        self._agent_algorithm.reset_states()

    def save_actor(self, path):
        self._agent_algorithm.save_actor(path)
