import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Concatenate, Add, Reshape, Lambda, Activation
from tensorflow.python.keras.initializers import RandomUniform
from .layer_norm import LayerNorm
from .noisy_dense import NoisyDense


def dense_block(input_layer, hiddens, activation='relu', 
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
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 layer_norm=False, noisy_layer=False,
                 output_activation=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.out_activation = output_activation
        self.scope = scope or 'CriticNetwork'
        self.model = self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        if self.act_insert_block == -1:
            model_inputs = [input_state]
        else:
            input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
            model_inputs = [input_state, input_action]

        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(out, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
            out = Dense(1, self.out_activation,
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
            return CriticNetwork(state_shape=self.state_shape,
                                 action_size=self.action_size,
                                 hiddens=self.hiddens,
                                 activations=self.activations,
                                 action_insert_block=self.act_insert_block,
                                 layer_norm=self.layer_norm,
                                 noisy_layer=self.noisy_layer,
                                 output_activation=self.out_activation,
                                 scope=scope)

    def get_info(self):
        info = {}
        info['architecture'] = 'standard'
        info['hiddens'] = self.hiddens
        info['activations'] = self.activations
        info['layer_norm'] = self.layer_norm
        info['noisy_layer'] = self.noisy_layer
        info['output_activation'] = self.out_activation
        info['action_insert_block'] = self.act_insert_block
        return info


class DuelingCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 layer_norm=False, noisy_layer=False,
                 output_activation=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.act_insert_block = action_insert_block
        self.out_activation = output_activation
        self.scope = scope or 'DuelingCriticNetwork'
        self.model = self.build_model()
        
    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(self.act_insert_block):
                out = dense_block(out, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
            val = out
            adv = Concatenate(axis=1)([out, input_action])
            for i in range(self.act_insert_block, len(self.hiddens)):
                val = dense_block(val, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
                adv = dense_block(adv, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
            val = Dense(1, self.out_activation,
                        kernel_initializer=RandomUniform(-3e-3, 3e-3),
                        bias_initializer=RandomUniform(-3e-3, 3e-3))(val)
            adv = Dense(1, self.out_activation,
                        kernel_initializer=RandomUniform(-3e-3, 3e-3),
                        bias_initializer=RandomUniform(-3e-3, 3e-3))(adv)
            out = Add()([val, adv])
            model = keras.models.Model(inputs=[input_state, input_action], outputs=out)
        return model

    def get_info(self):
        info = super(DuelingCriticNetwork, self).get_info()
        info['architecture'] = 'dueling'


class QuantileCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 num_atoms=50,
                 layer_norm=False, noisy_layer=False,
                 output_activation=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.act_insert_block = action_insert_block

        self.num_atoms = num_atoms
        tau_min = 1 / (2 * num_atoms) 
        tau_max = 1 - tau_min
        self.tau = tf.lin_space(start=tau_min, stop=tau_max, num=num_atoms)

        self.out_activation = output_activation
        self.scope = scope or 'QuantileCriticNetwork'
        self.model = self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(out, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
            atoms = Dense(self.num_atoms, self.out_activation,
                          kernel_initializer=RandomUniform(-3e-3, 3e-3),
                          bias_initializer=RandomUniform(-3e-3, 3e-3))(out)
            model = keras.models.Model(inputs=[input_state, input_action], outputs=atoms)
        return model

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return QuantileCriticNetwork(state_shape=self.state_shape,
                                         action_size=self.action_size,
                                         hiddens=self.hiddens,
                                         activations=self.activations,
                                         layer_norm=self.layer_norm,
                                         noisy_layer=self.noisy_layer,
                                         action_insert_block=self.act_insert_block,
                                         output_activation=self.out_activation,
                                         num_atoms=self.num_atoms,
                                         scope=scope)

    def get_info(self):
        info = super(QuantileCriticNetwork, self).get_info()
        info['architecture'] = 'quantile'
        info['num_atoms'] = self.num_atoms


class CategoricalCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 num_atoms=51, v=(-10., 10.),
                 layer_norm=False, noisy_layer=False,
                 output_activation=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.layer_norm = layer_norm
        self.noisy_layer = noisy_layer
        self.act_insert_block = action_insert_block

        self.num_atoms = num_atoms
        self.v_min, self.v_max = v
        self.delta_z = (self.v_max - self.v_min) / (num_atoms - 1)
        self.z = tf.lin_space(start=self.v_min, stop=self.v_max, num=num_atoms)

        self.out_activation = output_activation
        self.scope = scope or 'CategoricalCriticNetwork'
        self.model = self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(out, self.hiddens[i], self.activations[i],
                                  self.layer_norm, self.noisy_layer)
            logits = Dense(self.num_atoms, self.out_activation)(out)
            probs = Lambda(lambda x: tf.nn.softmax(x, axis=-1))(logits)
            model = keras.models.Model(inputs=[input_state, input_action], outputs=probs)
        return model

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return CategoricalCriticNetwork(state_shape=self.state_shape,
                                            action_size=self.action_size,
                                            hiddens=self.hiddens,
                                            activations=self.activations,
                                            layer_norm=self.layer_norm,
                                            noisy_layer=self.noisy_layer,
                                            action_insert_block=self.act_insert_block,
                                            output_activation=self.out_activation,
                                            num_atoms=self.num_atoms,
                                            v=(self.v_min, self.v_max),
                                            scope=scope)

    def get_info(self):
        info = super(CategoricalCriticNetwork, self).get_info()
        info['architecture'] = 'categorical'
        info['num_atoms'] = self.num_atoms
        info['domain'] = (self.v_min, self.v_max)
