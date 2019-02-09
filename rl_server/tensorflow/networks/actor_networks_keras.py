import importlib

import tensorflow as tf
from tensorflow.python import keras

from rl_server.tensorflow.networks.critic_networks_keras import process_layer_args

# from tensorflow.python.keras.layers import Dense, Reshape, Lambda, Activation
# from tensorflow.python.keras.initializers import RandomUniform

# from .layer_norm import LayerNorm
# from .noisy_dense import NoisyDense


# def dense_block(input_layer, hiddens, activation="relu",
#                 layer_norm=False, noisy_layer=False):
#     out = input_layer
#     if noisy_layer:
#         out = NoisyDense(hiddens, None)(out)
#     else:
#         out = Dense(hiddens, None)(out)
#     if layer_norm:
#         out = LayerNorm()(out)
#     out = Activation(activation)(out)
#     return out


class ActorNetwork:

    def __init__(self, state_shapes, action_size, nn_arch, scope=None):

        self.nn_arch = nn_arch
        self.state_shapes = state_shapes
        self.action_size = action_size
        self.scope = scope or "ActorNetwork"
        self.model = self.build_model()

    def build_model(self):

        with tf.variable_scope(self.scope):
            input_state = keras.layers.Input(shape=self.state_shapes[0], name="state_input")

            keras_module_layers = importlib.import_module('tensorflow.python.keras.layers')

            out_layer = input_state
            for layer_data in self.nn_arch:
                LayerClass = getattr(keras_module_layers, layer_data['type'])
                if 'args' in layer_data:
                    process_layer_args(layer_data['args'])
                    out_layer = LayerClass(**layer_data['args'])(out_layer)
                else:
                    out_layer = LayerClass()(out_layer)

            model = keras.models.Model(inputs=[input_state], outputs=out_layer)

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
        """copy only network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return ActorNetwork(
                state_shapes=self.state_shapes,
                action_size=self.action_size,
                nn_arch=self.nn_arch,
                scope=scope
            )

    def get_info(self):
        info = {}
        info["architecture"] = "standard"
        info["hiddens"] = self.hiddens
        info["activations"] = self.activations
        info["layer_norm"] = self.layer_norm
        info["noisy_layer"] = self.noisy_layer
        info["output_activation"] = self.out_activation
        return info
