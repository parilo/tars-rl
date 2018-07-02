import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Concatenate, Add, Reshape, Lambda
from tensorflow.python.keras.initializers import RandomUniform


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
                out = dense_block(out, self.hiddens[i], self.activations[i])
            out = Dense(1, self.out_activation)(out)
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
            model = keras.models.Model(inputs=[input_state, input_action], outputs=atoms)
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
                                         num_atoms=self.num_atoms,
                                         model=None,
                                         scope=scope)


class CategoricalCriticNetwork(CriticNetwork):

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 action_insert_block=0, num_atoms=51, v=(-10., 10.),
                 output_activation=None, model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.act_insert_block = action_insert_block

        self.num_atoms = num_atoms
        self.v_min, self.v_max = v
        self.delta_z = (self.v_max - self.v_min) / (num_atoms - 1)
        self.z = tf.lin_space(start=self.v_min, stop=self.v_max, num=num_atoms)

        self.out_activation = output_activation
        self.scope = scope or 'CategoricalCriticNetwork'
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
            logits = Dense(self.num_atoms, self.out_activation)(out)
            probs = Lambda(lambda x: tf.nn.softmax(x, axis=-1))(logits)
            model = keras.models.Model(inputs=[input_state, input_action], outputs=probs)
        return model

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return CategoricalCriticNetwork(state_shape=self.state_shape,
                                            action_size=self.action_size,
                                            hiddens=self.hiddens,
                                            activations=self.activations,
                                            action_insert_block=self.act_insert_block,
                                            output_activation=self.out_activation,
                                            num_atoms=self.num_atoms,
                                            v=(self.v_min, self.v_max),
                                            model=None,
                                            scope=scope)
