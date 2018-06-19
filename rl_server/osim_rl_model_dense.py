import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Concatenate


class DenseNetwork:

    def __init__(self, input_shapes, output_size, fully_connected=[400, 300],
                 activation='tanh', output_activation=None, model=None, scope=None):
        """
        Class for dense neural network which estimates either
        value function (critic mode) or policy (actor mode).

        Parameters
        ----------
        input_shapes: list
            actor mode -- [(state_size, )]
            critic mode -- [[(state_size, )], [(action_size, )]]
        output_size: int
            actor mode -- action space dimentionality
            critic mode -- value estimator dimensionality (usually `1` for value
                           function and `num_atoms` for value distribution)
        fully_connected: list of ints
            list of dense layers' parameters of the form [num_outputs, ...]
        activation: str
            activation function in keras
        scope: str
            unique name of a specific network
        """

        self._output_size = output_size
        self._fc = fully_connected
        self._activation = activation
        state_input = keras.layers.Input(shape=input_shapes[0], name='state_input_dummy')

        if len(input_shapes) == 1:
            self._scope = scope or 'ActorNetwork'
            self._input_shape = input_shapes[0]
            model_inputs = [state_input]
            input_layer = state_input
            output_activation = 'sigmoid'

        elif len(input_shapes) == 2:
            self._scope = scope or 'CriticNetwork'
            self._input_shape = (input_shapes[0][0] + input_shapes[1][0], )
            action_input = keras.layers.Input(shape=input_shapes[1], name='action_input_dummy')
            model_inputs = [state_input, action_input]
            input_layer = Concatenate(axis=1)(model_inputs)

        if model is None:
            with tf.variable_scope(self._scope):
                model_outputs = self.ff_network(output_activation)(input_layer)
                self._model = keras.models.Model(inputs=model_inputs, outputs=model_outputs)
        else:
            self._model = model

    def __call__(self, inputs):
        """
        Build neural network graph for actor or critic

        Parameters
        ----------
        inputs: list
            actor mode -- [states]
            critic mode -- [states, actions]

        In general the list of states consists of several parts which
        correspond to observations of different nature, e.g. images and vectors.
        """

        if len(inputs) == 2:
            return self._model([inputs[0][0], inputs[1]])
        else:
            return self._model(inputs)

    def variables(self):
        return self._model.trainable_weights

    def ff_network(self, output_activation=None):
        model = Sequential()
        model.add(Dense(self._fc[0], activation=self._activation, input_shape=self._input_shape))
        for num_units in self._fc[1:]:
            model.add(Dense(num_units, activation=self._activation))
        model.add(Dense(self._output_size, activation=output_activation))
        return model
