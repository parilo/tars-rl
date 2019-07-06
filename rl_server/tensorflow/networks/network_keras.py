import importlib
import copy

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Concatenate, Reshape, Activation, Lambda


def process_layer_args(layer_args):
    layer_args_copy = copy.deepcopy(layer_args)
    for arg_name, arg_val in layer_args_copy.items():
        if isinstance(arg_val, dict) and 'class' in arg_val:
            ArgClass = getattr(
                importlib.import_module(arg_val['module']),
                arg_val['class']
            )
            if 'args' in arg_val:
                arg_instance = ArgClass(**arg_val['args'])
            else:
                arg_instance = ArgClass()

            layer_args[arg_name] = arg_instance


def process_special_layers(layer_data, layer_input):
    layer_type = layer_data['type']
    if layer_type == 'Scale':
        return Lambda(lambda x: x * layer_data['mult'] + layer_data['bias'])(layer_input)
    else:
        return None


class NetworkKeras:

    def __init__(
        self,
        state_shapes,
        action_size,
        nn_arch,
        have_action_input=False,
        num_atoms=1,
        v=(-10., 10.),
        scope=None
    ):

        self.state_shapes = state_shapes
        self.action_size = action_size
        self.nn_arch = nn_arch
        self.has_action_input = have_action_input
        self.num_atoms = num_atoms
        self.v = v
        self.scope = scope or "CriticNetwork"
        self.model = self.build_model()

    def process_layers(self, layers_info, layers_input):

        print('--- branch')
        keras_module = importlib.import_module('tensorflow.python.keras.layers')

        out_layer = layers_input
        for layer_i, layer_data in enumerate(layers_info):

            special_layer_output = process_special_layers(layer_data, out_layer)

            if special_layer_output is None:
                LayerClass = getattr(keras_module, layer_data['type'])
                if 'args' in layer_data:
                    process_layer_args(layer_data['args'])
                    out_layer = LayerClass(**layer_data['args'])(out_layer)
                else:
                    out_layer = LayerClass()(out_layer)
            else:
                out_layer = special_layer_output

            print(out_layer)
        return out_layer

    def build_model(self):

        print('--- build', self.scope)

        with tf.variable_scope(self.scope):

            if 'fixed_batch_size' in self.nn_arch:
                input_state = [keras.layers.Input(
                    batch_shape=[self.nn_arch['fixed_batch_size']] + list(state_part),
                    name="state_input_" + str(i)
                ) for i, state_part in enumerate(self.state_shapes)]
            else:
                input_state = [
                    keras.layers.Input(
                        shape=state_part,
                        name="state_input_" + str(i)
                    )
                    for i, state_part in enumerate(self.state_shapes)
                ]

            model_inputs = input_state

            if self.has_action_input:
                model_inputs.append(
                    keras.layers.Input(shape=(self.action_size,), name="action_input")
                )

            if isinstance(self.nn_arch, list):

                branches_inputs = dict(zip(list(range(len(input_state))), input_state))

                if self.has_action_input:
                    branches_inputs['action'] = model_inputs[-1]

                for branch in self.nn_arch:
                    branch_input = branch['input']
                    if isinstance(branch_input, list):
                        branch_input = [branches_inputs[branch_input_part] for branch_input_part in branch_input]
                    else:
                        branch_input = branches_inputs[branch_input]

                    branch_output = self.process_layers(
                        branch['layers'],
                        branch_input
                    )
                    branches_inputs[branch['name']] = branch_output

                out_layer = branches_inputs['output']

            else:
                out_layer = self.process_layers(self.nn_arch['layers'], input_state[0])

            model = keras.models.Model(inputs=model_inputs, outputs=out_layer)

        return model

    def reset_states(self):
        self.model.reset_states()

    def get_input_size(self, shape):
        if len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape[0] * shape[1]

    def __call__(self, inputs):
        if self.has_action_input:
            model_input = inputs[0] + [inputs[1]]
        else:
            model_input = inputs
        return self.model(model_input)

    def variables(self):
        return self.model.trainable_weights

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return NetworkKeras(
                state_shapes=self.state_shapes,
                action_size=self.action_size,
                nn_arch=self.nn_arch,
                num_atoms=self.num_atoms,
                v=self.v,
                have_action_input=self.has_action_input,
                scope=scope)