import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Reshape, Lambda, Activation
from tensorflow.python.keras.initializers import RandomUniform
from .layer_norm import LayerNorm


def dense_block(input_layer, hiddens, activation='relu', layer_norm=False):
    out = input_layer 
    for num_units in hiddens:
        out = Dense(num_units, None)(out)
        if layer_norm:
            out = LayerNorm()(out)
        out = Activation(activation)(out)
    return out


class ActorNetwork:

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 layer_norm=False, output_activation=None,
                 model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.layer_norm = layer_norm
        self.out_activation = output_activation
        self.scope = scope or 'ActorNetwork'
        self.model = model or self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                out = dense_block(out, self.hiddens[i], self.activations[i], self.layer_norm)
            out = Dense(self.action_size, self.out_activation,
                        kernel_initializer=RandomUniform(-3e-3, 3e-3),
                        bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
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
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return ActorNetwork(state_shape=self.state_shape,
                                action_size=self.action_size,
                                hiddens=self.hiddens,
                                activations=self.activations,
                                layer_norm=self.layer_norm,
                                output_activation=self.out_activation,
                                model=None,
                                scope=scope)


class GMMActorNetwork(ActorNetwork):
    
    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 num_components=1,
                 layer_norm=False, output_activation=None,
                 model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.K = num_components
        self.layer_norm = layer_norm
        self.out_activation = output_activation
        self.scope = scope or 'GMMActorNetwork'
        self.model = model or self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                out = dense_block(out, self.hiddens[i], self.activations[i], self.layer_norm)

            log_weight = Dense(self.K, None,
                               kernel_initializer=RandomUniform(-3e-3, 3e-3),
                               bias_initializer=RandomUniform(-3e-3, 3e-3))(out)

            mu = Dense(self.K * self.action_size, None,
                       kernel_initializer=RandomUniform(-3e-3, 3e-3),
                       bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
            mu = Reshape((self.K, self.action_size))(mu)

            log_std = Dense(self.K * self.action_size, None,
                            kernel_initializer=RandomUniform(-3e-3, 3e-3),
                            bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
            log_std = Reshape((self.K, self.action_size))(log_std)

            model = keras.models.Model(inputs=[input_state], outputs=[log_weight, mu, log_std])
        return model

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return GMMActorNetwork(state_shape=self.state_shape,
                                   action_size=self.action_size,
                                   hiddens=self.hiddens,
                                   activations=self.activations,
                                   layer_norm=self.layer_norm,
                                   num_components=self.K,
                                   output_activation=self.out_activation,
                                   model=None,
                                   scope=scope)
