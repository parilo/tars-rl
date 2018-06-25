import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape, \
    Flatten, Dropout, Conv2D, MaxPooling2D, Concatenate


class LunarLanderModelDense(object):
    def __init__(self, input_shapes, output_size, model=None, scope=None):

        self._input_shapes = input_shapes
        self._output_size = output_size
        self._scope = scope or 'LunarLanderModelDense'

        input_shape = input_shapes[0]

        if model is None:

            state_input = keras.layers.Input(
                shape=input_shape,
                name='state_input_dummy'
            )

            with tf.variable_scope(self._scope):

                if self._output_size == 1:
                    # critic
                    critic_shape = input_shapes[1]
                    critic_input = keras.layers.Input(
                        shape=critic_shape,
                        name='critic_input_dummy'
                    )

                    concatenated_input = Concatenate(axis=1)([
                        Reshape((-1,))(state_input),
                        critic_input
                    ])
                    concatenated_input_shape = (
                        input_shape[0]*input_shape[1] + critic_shape[0],
                    )

                    ff_network = Sequential([
                        Dense(
                            64,
                            activation='relu',
                            input_shape=concatenated_input_shape
                        ),
                        Dense(64, activation='relu'),
                        Dense(self._output_size)
                    ])

                    model_inputs = [
                        state_input,
                        critic_input
                    ]

                    self._model = keras.models.Model(
                        inputs=model_inputs,
                        outputs=ff_network(concatenated_input)
                    )
                else:
                    # actor
                    ff_network = Sequential([
                        Reshape((-1,), input_shape=input_shape),
                        Dense(32, activation='tanh', name='critic_1'),
                        Dense(32, activation='tanh', name='critic_2'),
                        Dense(self._output_size, activation='tanh', name='critic_output')
                    ])

                    model_inputs = [
                        state_input
                    ]

                    self._model = keras.models.Model(
                        inputs=model_inputs,
                        outputs=ff_network(state_input)
                    )

        else:
            self._model = model

    def __call__(self, xs):
        if len(xs) == 2:
            # critic
            return self._model([xs[0][0], xs[1]])
        else:
            # actor
            return self._model(xs)

    def variables(self):
        return self._model.trainable_weights

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope) as sc:
            m = keras.models.model_from_json(self._model.to_json())
            m.set_weights(self._model.get_weights())
            return LunarLanderModelDense(
                input_shapes=self._input_shapes,
                output_size=self._output_size,
                model=m,
                scope=scope
            )
