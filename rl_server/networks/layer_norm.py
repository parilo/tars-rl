from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class LayerNorm(Layer):

    def __init__(self, axis=-1, eps=1e-6, **kwargs):
        self.axis = axis
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer='zeros',
                                    trainable=True)
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=self.axis, keepdims=True)
        std = K.std(x, axis=self.axis, keepdims=True)
        out = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return out

    def compute_output_shape(self, input_shape):
        return input_shape
