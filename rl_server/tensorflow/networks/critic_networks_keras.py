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


class CriticNetwork:

    def __init__(
        self,
        state_shapes,
        action_size,
        nn_arch,
        use_action_input=False,
        use_next_state_input=False,
        # action_insert_block=0,
        # num_atoms=1,
        # v=(-10., 10.),
        scope=None
    ):
        self.state_shapes = state_shapes
        self.action_size = action_size
        self.use_action_input = use_action_input
        self.use_next_state_input = use_next_state_input
        self.nn_arch = nn_arch
        # self.action_insert_block = action_insert_block
        # self.num_atoms = num_atoms
        # self.v = v
        self.scope = scope or "CriticNetwork"
        self.model = self.build_model()

    def process_layers(self, layers_info, layers_input):

        # print('--- branch')
        keras_module = importlib.import_module('tensorflow.python.keras.layers')

        out_layer = layers_input
        for layer_i, layer_data in enumerate(layers_info):

            special_layer_output = process_special_layers(layer_data, out_layer)

            if special_layer_output is None:
                LayerClass = getattr(keras_module, layer_data['type'])
                # if layer_i == self.action_insert_block:
                #     out_layer = Concatenate(axis=1)([out_layer, input_action])
                if 'args' in layer_data:
                    process_layer_args(layer_data['args'])
                    # print('--- input', out_layer)
                    out_layer = LayerClass(**layer_data['args'])(out_layer)
                else:
                    out_layer = LayerClass()(out_layer)
            else:
                out_layer = special_layer_output

            # print(out_layer)
        return out_layer

    def create_state_input(self, prefix='state_input'):
        if 'fixed_batch_size' in self.nn_arch:
            input_state = [keras.layers.Input(
                batch_shape=[self.nn_arch['fixed_batch_size']] + list(state_part),
                name=prefix + "_" + str(i)
            ) for i, state_part in enumerate(self.state_shapes)]
        else:
            input_state = [
                keras.layers.Input(
                    shape=state_part,
                    name=prefix + "_" + str(i)
                )
                for i, state_part in enumerate(self.state_shapes)
            ]
        return input_state

    def build_model(self):

        with tf.variable_scope(self.scope):

            input_state = self.create_state_input()

            if self.use_action_input:
                input_action = keras.layers.Input(shape=(self.action_size, ), name="action_input")

            if self.use_next_state_input:
                input_next_state = self.create_state_input('next_state_input')

            if isinstance(self.nn_arch, list):

                branches_inputs_ph = input_state + (input_next_state if self.use_next_state_input else [])
                branches_inputs = dict(zip(list(range(len(branches_inputs_ph))), branches_inputs_ph))
                if self.use_action_input:
                    branches_inputs['input_action'] = input_action

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
                raise NotImplementedError()

            model_inputs = input_state + \
                           ([input_action] if self.use_action_input else []) + \
                           (input_next_state if self.use_next_state_input else [])
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
        print('--- inputs', inputs)
        # if self.action_insert_block == -1:
        # if not self.use_action_input:
        #     # state_input = inputs[0]
        #     # model_input = state_input
        #     model_input = inputs
        # else:
        #     state_input = inputs[0][0]
        #     action_input = inputs[1]
        #     model_input = [state_input, action_input]
        # return self.model(model_input)
        return self.model(inputs)

    def variables(self):
        return self.model.trainable_weights

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return CriticNetwork(
                state_shapes=self.state_shapes,
                action_size=self.action_size,
                use_action_input=self.use_action_input,
                use_next_state_input=self.use_next_state_input,
                nn_arch=self.nn_arch,
                # action_insert_block=self.action_insert_block,
                # num_atoms=self.num_atoms,
                # v=self.v,
                scope=scope)

    # def get_info(self):
    #     info = {}
    #     info["architecture"] = "standard"
    #     info["hiddens"] = self.hiddens
    #     info["activations"] = self.activations
    #     info["layer_norm"] = self.layer_norm
    #     info["noisy_layer"] = self.noisy_layer
    #     info["output_activation"] = self.out_activation
    #     info["action_insert_block"] = self.act_insert_block
    #     info["num_atoms"] = self.num_atoms
    #     return info
