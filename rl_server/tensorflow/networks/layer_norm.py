from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class LayerNorm(Layer):

    def __init__(self, axis=-1, eps=1e-6, **kwargs):
        self.axis = axis
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[self.axis]
        self.gamma = self.add_weight(shape=(input_dim,), initializer='ones', name='gamma')
        self.beta = self.add_weight(shape=(input_dim,), initializer='zeros', name='beta')
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=self.axis, keepdims=True)
        std = K.std(x, axis=self.axis, keepdims=True)
        out = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return out

    def compute_output_shape(self, input_shape):
        return input_shape
