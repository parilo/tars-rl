import tensorflow as tf

from rl_server.server.rl_trainer_tf import make_session
from rl_server.tensorflow.algo.algo_fabric import create_algorithm


class AgentModel:
    
    def __init__(self, exp_config, rl_client):
        self._rl_client = rl_client
        self._agent_algorithm = create_algorithm(exp_config)
        self._sess = make_session(num_cpu=1)

    def set_weights(self, weights):
        self._agent_algorithm.set_weights(self._sess, weights)
        
    def act_batch(self, states, mode="default"):
        if mode == "default":
            actions = self._agent_algorithm.act_batch(self._sess, states)
        elif mode == "deterministic":
            actions = self._agent_algorithm.act_batch_deterministic(self._sess, states)
        elif mode == "with_gradients":
            # actually, here it will return actions and grads
            actions = self._agent_algorithm.act_batch_with_gradients(self._sess, states)
        else:
            raise NotImplementedError
        return actions

    def act_batch_target(self, states):
        return self._agent_algorithm.act_batch_target(self._sess, states)

    def fetch(self, index=0):
        self.set_weights(self._rl_client.get_weights(index=index))

    def load_checkpoint(self, path):

        reader = tf.train.NewCheckpointReader(path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        # this may cause problems
        # when there are variables with :1, etc.
        # suffixes in checkpoint
        var_dict = dict(zip(
            [v_name + ':0' for v_name in var_to_shape_map.keys()],
            [True] * len(var_to_shape_map)
        ))

        var_list = []
        for v in tf.global_variables():
            if v.name in var_dict:
                var_list.append(v)

        self._saver = tf.train.Saver(max_to_keep=None, var_list=var_list)
        self._saver.restore(self._sess, path)

        unint_var_names = list(self._sess.run(tf.report_uninitialized_variables()))
        unint_var_names = [v_name.decode('utf-8') for v_name in unint_var_names]
        unint_vars = [v for v in tf.global_variables() if v.name.split(':')[0] in set(unint_var_names)]

        if len(unint_vars) > 0:
            print(
                '--- warning: there are uninitialized vars which will be initialized with default values',
                unint_vars
            )

        self._sess.run(tf.variables_initializer(unint_vars))

    def reset_states(self):
        self._agent_algorithm.reset_states()

    def save_actor(self, path):
        self._agent_algorithm.save_actor(self._sess, path)
