import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import (
    Dense, Concatenate, Reshape, Activation)
from tensorflow.python.keras.initializers import RandomUniform
from .layer_norm import LayerNorm
from .noisy_dense import NoisyDense


def dense_block(input_layer, hiddens, activation="relu",
                layer_norm=False, noisy_layer=False):
    out = input_layer
    for num_units in hiddens:
        if noisy_layer:
            out = NoisyDense(num_units, None)(out)
        else:
            out = Dense(num_units, None)(out)
        if layer_norm:
            out = LayerNorm()(out)
        out = Activation(activation)(out)
    return out


class CriticNetwork:

    def __init__(self, state_shape, action_size,
                 hiddens=[[256, 128], [64, 32]],
                 activations=["relu", "tanh"],
                 action_insert_block=0, num_atoms=1,
                 v=(-10., 10.),
                 layer_norm=False, noisy_layer=False,
                 output_activation=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block
        self.num_atoms = num_atoms
        self.v = v
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.out_activation = output_activation
        self.scope = scope or "CriticNetwork"
        self.model = self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(
            shape=self.state_shape, name="state_input")
        if self.act_insert_block == -1:
            model_inputs = [input_state]
        else:
            input_action = keras.layers.Input(
                shape=(self.action_size, ), name="action_input")
            model_inputs = [input_state, input_action]

        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(
                    out, self.hiddens[i], self.activations[i],
                    self.layer_norm, self.noisy_layer)
            out = Dense(
                self.num_atoms, self.out_activation,
                kernel_initializer=RandomUniform(-3e-3, 3e-3),
                bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
            model = keras.models.Model(inputs=model_inputs, outputs=out)
        return model

    def get_input_size(self, shape):
        if len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape[0] * shape[1]

    def __call__(self, inputs):
        if self.act_insert_block == -1:
            state_input = inputs[0]
            model_input = state_input
        else:
            state_input = inputs[0][0]
            action_input = inputs[1]
            model_input = [state_input, action_input]
        return self.model(model_input)

    def variables(self):
        return self.model.trainable_weights

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return CriticNetwork(
                state_shape=self.state_shape,
                action_size=self.action_size,
                hiddens=self.hiddens,
                activations=self.activations,
                action_insert_block=self.act_insert_block,
                num_atoms=self.num_atoms,
                v=self.v,
                layer_norm=self.layer_norm,
                noisy_layer=self.noisy_layer,
                output_activation=self.out_activation,
                scope=scope)

    def get_info(self):
        info = {}
        info["architecture"] = "standard"
        info["hiddens"] = self.hiddens
        info["activations"] = self.activations
        info["layer_norm"] = self.layer_norm
        info["noisy_layer"] = self.noisy_layer
        info["output_activation"] = self.out_activation
        info["action_insert_block"] = self.act_insert_block
        info["num_atoms"] = self.num_atoms
        return info
