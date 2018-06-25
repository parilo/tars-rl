import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Concatenate, BatchNormalization, Add, Reshape


def dense_block(input_layer, hiddens, activation='relu'):
    out = input_layer 
    for num_units in hiddens:
        out = Dense(num_units, activation)(out)
    return out


class ActorNetwork:

    def __init__(self, state_shape, action_size,
                 hiddens = [[256, 128], [64, 32]],
                 activations=['relu', 'tanh'],
                 output_activation=None, model=None, scope=None):

        self.state_shape = state_shape
        self.action_size = action_size
        self.hiddens = hiddens
        self.activations = activations
        self.out_activation = output_activation
        self.scope = scope or 'ActorNetwork'
        self.model = model or self.build_model()

    def build_model(self):
        input_state = keras.layers.Input(shape=self.state_shape, name='state_input')
        input_size = self.get_input_size(self.state_shape)
        out = Reshape((input_size, ))(input_state)
        with tf.variable_scope(self.scope):
            for i in range(len(self.hiddens)):
                out = dense_block(out, self.hiddens[i], self.activations[i])
            out = Dense(self.action_size, self.out_activation)(out)
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
            model = keras.models.model_from_json(self.model.to_json())
            model.set_weights(self.model.get_weights())
            return ActorNetwork(state_shape=self.state_shape,
                                action_size=self.action_size,
                                hiddens=self.hiddens,
                                activations=self.activations,
                                output_activation=self.out_activation,
                                model=model,
                                scope=scope)
