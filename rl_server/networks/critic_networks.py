import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Concatenate, Add, Reshape, Lambda


def dense_block(input_layer, hiddens, activation='relu'):
    out = input_layer 
    for num_units in hiddens:
        out = Dense(num_units, activation)(out)
    return out


class CriticNetwork:

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 output_activation=None, model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block
        self.out_activation = output_activation
        self.scope = scope or 'CriticNetwork'
        self.model = model or self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(out, self.hiddens[i], self.activations[i])
            out = Dense(1, self.out_activation)(out)
            model = keras.models.Model(inputs=[input_state, input_action], outputs=out)
        return model

    def get_input_size(self, shape):
        if len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape[0] * shape[1]

    def __call__(self, inputs):
        state_input = inputs[0][0]
        action_input = inputs[1]
        return self.model([state_input, action_input])

    def variables(self):
        return self.model.trainable_weights

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            model = keras.models.model_from_json(self.model.to_json())
            model.set_weights(self.model.get_weights())
            return CriticNetwork(state_shape=self.state_shape,
                                 action_size=self.action_size,
                                 hiddens=self.hiddens,
                                 activations=self.activations,
                                 action_insert_block=self.act_insert_block,
                                 output_activation=self.out_activation,
                                 model=model,
                                 scope=scope)


class DuelingCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0,
                 output_activation=None, model=None, scope=None):
        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block
        self.out_activation = output_activation
        self.scope = scope or 'DuelingCriticNetwork'
        self.model = model or self.build_model()
        
    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(self.act_insert_block):
                out = dense_block(out, self.hiddens[i], self.activations[i])
            val = out
            adv = Concatenate(axis=1)([out, input_action])
            for i in range(self.act_insert_block, len(self.hiddens)):
                val = dense_block(val, self.hiddens[i], self.activations[i])
                adv = dense_block(adv, self.hiddens[i], self.activations[i])
            val = Dense(1, self.out_activation)(val)
            adv = Dense(1, self.out_activation)(adv)
            out = Add()([val, adv])
            model = keras.models.Model(inputs=[input_state, input_action], outputs=out)
        return model


class QuantileCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0, num_atoms=50,
                 output_activation=None, model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block
        self.num_atoms = num_atoms
        tau_min = 1 / (2 * num_atoms) 
        tau_max = 1 - tau_min
        self.tau = tf.lin_space(start=tau_min, stop=tau_max, num=num_atoms)
        self.out_activation = output_activation
        self.scope = scope or 'QuantileCriticNetwork'
        self.model = model or self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_action = keras.layers.Input(shape=(self.action_size, ), name='action_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                if (i == self.act_insert_block):
                    out = Concatenate(axis=1)([out, input_action])
                out = dense_block(out, self.hiddens[i], self.activations[i])
            atoms = Dense(self.num_atoms, self.out_activation)(out)
            q_values = Lambda(lambda x: tf.reduce_sum(x, axis=-1))(atoms)
            model = keras.models.Model(inputs=[input_state, input_action], outputs=[atoms, q_values])
        return model

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return QuantileCriticNetwork(state_shape=self.state_shape,
                                         action_size=self.action_size,
                                         hiddens=self.hiddens,
                                         activations=self.activations,
                                         action_insert_block=self.act_insert_block,
                                         output_activation=self.out_activation,
                                         num_atoms = self.num_atoms,
                                         model=None,
                                         scope=scope)
