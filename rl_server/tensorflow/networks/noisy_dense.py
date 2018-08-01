import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.initializers import RandomUniform


class NoisyDense(Layer):

    def __init__(self, units,
                 activation=None,
                 factorised=True,
                 use_bias=True,
                 **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.factorised = factorised
        self.use_bias = use_bias

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        if self.factorised:
            init_mu = 1. / np.sqrt(self.input_dim.value)
            init_sig = 0.5 / np.sqrt(self.input_dim.value)
        else:
            init_mu = np.sqrt(3. / self.input_dim.value)
            init_sig = 0.017

        self.kernel_mu = self.add_weight(shape=(self.input_dim, self.units),
                                         initializer=RandomUniform(-init_mu, init_mu),
                                         name='kernel_mu')
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=RandomUniform(-init_sig, init_sig),
                                            name='kernel_sigma')

        if self.use_bias:
            self.bias_mu = self.add_weight(shape=(self.units,),
                                           initializer=RandomUniform(-init_mu, init_mu),
                                           name='bias_mu')
            self.bias_sigma = self.add_weight(shape=(self.units,),
                                              initializer=RandomUniform(-init_sig, init_sig),
                                              name='bias_sigma')

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def scale_noise(self, noise):
        return K.sign(noise) * K.sqrt(K.abs(noise))

    def call(self, inputs):

        if self.factorised:
            noise_in = self.scale_noise(K.random_normal(shape=(self.input_dim.value,)))
            noise_out = self.scale_noise(K.random_normal(shape=(self.units,)))
            kernel_noise = noise_in[:, None] * noise_out[None, :]
            bias_noise = noise_out
        else:
            kernel_noise = K.random_normal(shape=(self.input_dim.value, self.units))
            bias_noise = K.random_normal(shape=(self.units,))

        out = K.dot(inputs, self.kernel_mu + self.kernel_sigma * kernel_noise)
        if self.use_bias:
            out = K.bias_add(out, self.bias_mu + self.bias_sigma * bias_noise,
                             data_format='channels_last')
        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
