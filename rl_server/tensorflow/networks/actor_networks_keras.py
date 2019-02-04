import importlib

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Reshape, Lambda, Activation
from tensorflow.python.keras.initializers import RandomUniform

from .layer_norm import LayerNorm
from .noisy_dense import NoisyDense


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

        input_state = keras.layers.Input(shape=self.state_shapes[0], name="state_input")

        keras_module = importlib.import_module('tensorflow.python.keras.layers')
        out_layer = input_state
        for layer_data in self.nn_arch:
            LayerClass = getattr(keras_module, layer_data['type'])
            if 'args' in layer_data:
                out_layer = LayerClass(**layer_data['args'])(out_layer)
            else:
                out_layer = LayerClass()(out_layer)
            print('layer', out_layer)

        return keras.models.Model(inputs=[input_state], outputs=out_layer)

        # input_state = keras.layers.Input(shape=self.state_shape, name="state_input")
        # input_size = self.get_input_size(self.state_shape)
        # out = Reshape((input_size, ))(input_state)
        # with tf.variable_scope(self.scope):
        #     for i in range(len(self.hiddens)):
        #         out = dense_block(out, self.hiddens[i], self.activations[i],
        #                           self.layer_norm, self.noisy_layer)
        #     out = Dense(self.action_size, self.out_activation,
        #                 kernel_initializer=RandomUniform(-3e-3, 3e-3),
        #                 bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
        #     model = keras.models.Model(inputs=[input_state], outputs=out)

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
