import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Reshape, Lambda, Activation, LSTM
from tensorflow.python.keras.initializers import RandomUniform

from .layer_norm import LayerNorm
from .noisy_dense import NoisyDense


def dense_block(input_layer, num_units, activation="relu",
                layer_norm=False, noisy_layer=False):
    out = input_layer
    if noisy_layer:
        out = NoisyDense(num_units, None)(out)
    else:
        out = Dense(num_units, None)(out)
    if layer_norm:
        out = LayerNorm()(out)
    out = Activation(activation)(out)
    return out


class ActorNetwork:

    def __init__(self, state_shape, action_size,
                 embedding_layers=[256, 128, 64, 32],
                 embedding_activations=["relu", "relu", "relu", "relu"],
                 lstm_layers=[64],
                 lstm_activations=["relu"],
                 output_layers=[256, 128, 64, 32],
                 output_layers_activations=["relu", "relu", "relu", "relu"],
                 output_activation="tanh",
                 layer_norm=False,
                 noisy_layer=False,
                 scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.embedding_layers = embedding_layers
        self.embedding_activations = embedding_activations
        self.lstm_layers = lstm_layers
        self.lstm_activations = lstm_activations
        self.output_layers = output_layers
        self.output_layers_activations = output_layers_activations
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.scope = scope or "ActorNetwork"
        self.model = self.build_model()

    def build_model(self):

        with tf.variable_scope(self.scope):
            input_state = keras.layers.Input(shape=self.state_shape, name="state_input")
            out = input_state
            for layer_size, activation in zip(self.embedding_layers, self.embedding_activations):
                out = dense_block(
                    out,
                    layer_size,
                    activation,
                    self.layer_norm,
                    self.noisy_layer
                )

            for layer_size, activation in zip(self.lstm_layers[:-1], self.lstm_activations[:-1]):
                out = LSTM(layer_size, activation=activation, return_sequences=True)(out)
            out = LSTM(
                units=self.lstm_layers[-1],
                activation=self.lstm_activations[-1]
            )(out)

            for layer_size, activation in zip(self.output_layers, self.output_layers_activations):
                out = dense_block(
                    out,
                    layer_size,
                    activation,
                    self.layer_norm,
                    self.noisy_layer
                )

            out = Dense(
                self.action_size,
                self.output_activation,
                kernel_initializer=RandomUniform(-3e-3, 3e-3),
                bias_initializer=RandomUniform(-3e-3, 3e-3)
            )(out)

            model = keras.models.Model(inputs=[input_state], outputs=out)
        return model

    def get_input_size(self, shape):
        if len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape[0] * shape[1]

    def __call__(self, inputs):
        if isinstance(inputs, list):
            state_input = inputs[0]
        else:
            state_input = inputs
        return self.model(state_input)

    def variables(self):
        return self.model.trainable_weights

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return ActorNetwork(state_shape=self.state_shape,
                                action_size=self.action_size,
                                embedding_layers=self.embedding_layers,
                                embedding_activations=self.embedding_activations,
                                lstm_layers=self.lstm_layers,
                                lstm_activations=self.lstm_activations,
                                output_layers=self.output_layers,
                                output_layers_activations=self.output_layers_activations,
                                layer_norm=self.layer_norm,
                                noisy_layer=self.noisy_layer,
                                output_activation=self.output_activation,
                                scope=scope)

    def get_info(self):
        info = {}
        info["architecture"] = "standard"
        info["embedding_layers"] = self.embedding_layers
        info["embedding_activations"] = self.embedding_activations
        info["lstm_layers"] = self.lstm_layers
        info["lstm_activations"] = self.lstm_activations
        info["output_layers"] = self.output_layers
        info["output_layers_activations"] = self.output_layers_activations
        info["layer_norm"] = self.layer_norm
        info["noisy_layer"] = self.noisy_layer
        info["output_activation"] = self.output_activation
        return info
