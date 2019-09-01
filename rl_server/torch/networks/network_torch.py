import importlib

import torch.nn as nn
import numpy as np


def _calc_branch(branch, branch_input_values):
    x = branch_input_values
    print('--- _calc_branch', x)
    for layer in branch['layers']:
        x = layer(x)
    return x


def _calc_branch_with_dependencies(branches, branch_name, x):
    print('--- _calc_branch_with_dependencies', branch_name)

    branch_input = branches[branch_name]['input']
    if not isinstance(branch_input, list):
        branch_input = [branch_input]

    branch_input_values = []
    for branch_input_name in branch_input:
        if isinstance(branch_input_name, int):
            print('--- input branch is int')
            branch_input_values.append(x[branch_input_name])
        else:
            print('--- input branch is not int')
            branch_input_values.append(_calc_branch_with_dependencies(branches, branch_input_name, x))

    if len(branch_input_values) == 1:
        branch_input_values = branch_input_values[0]
    return _calc_branch(branches[branch_name], branch_input_values)


class LayerBase:

    def get_module(self):
        return None


class TorchLayer(LayerBase):

    def __init__(self, layer):
        self._layer = layer

    def __call__(self, x):
        return self._layer(x)

    def get_module(self):
        return self._layer


class FlattenLayer(LayerBase):

    def __call__(self, x):
        if len(x.shape) > 1:
            return x.view(x.shape[0], np.product(x.shape[1:]))
        else:
            return x


def process_special_layers(layer_data):
    layer_type = layer_data['type']
    if layer_type == 'Flatten':
        return FlattenLayer()
    else:
        return None


class NetworkTorch(nn.Module):

    def __init__(
        self,
        nn_arch,
        have_action_input=False,
        use_next_state_input=False,
        num_atoms=1,
        # v=(-10., 10.),
        # scope=None
    ):
        super().__init__()
        self.nn_arch = nn_arch
        self.has_action_input = have_action_input
        self.use_next_state_input = use_next_state_input
        self.num_atoms = num_atoms
        # self.v = v
        # self.scope = scope or "CriticNetwork"
        self._branches = self.build_model()

    def process_layers(self, layers_info):

        print('--- branch')
        nn_module = importlib.import_module('torch.nn')

        layers = []

        # nn.Linear
        # nn.ReLU

        for layer_i, layer_data in enumerate(layers_info):

            layer = process_special_layers(layer_data)

            if layer is None:
                LayerClass = getattr(nn_module, layer_data['type'])
                if 'args' in layer_data:
                    # process_layer_args(layer_data['args'])
                    torch_layer_class_instance = LayerClass(**layer_data['args'])
                else:
                    torch_layer_class_instance = LayerClass()
                layer = TorchLayer(torch_layer_class_instance)

            layers.append(layer)

        return layers

    def build_model(self):

        branches = {}

        if isinstance(self.nn_arch, list):

            for branch in self.nn_arch:

                branch_layers = self.process_layers(branch['layers'])
                branches[branch['name']] = {
                    'layers': branch_layers,
                    'input': branch['input']
                }

        else:
            layers = self.process_layers(self.nn_arch['layers'], input_state[0])
            branches['output'] = {
                'layers': layers,
                'input': 0
            }

        for name, branch_info in branches.items():
            for i, layer in enumerate(branch_info['layers']):
                self.add_module('{}_layer_{}'.format(name, i), layer.get_module())

        return branches

    def forward(self, x):
        """
        :param x: list
        :return:
        """

        if 'output' not in self._branches:
            raise RuntimeError(
                'you must have branch with name "output" in nn_arch in the config'
                'or not to use branches'
            )

        return _calc_branch_with_dependencies(self._branches, 'output', x)

    def reset_states(self):
        pass

    # def get_input_size(self, shape):
    #     if len(shape) == 1:
    #         return shape[0]
    #     elif len(shape) == 2:
    #         return shape[0] * shape[1]

    # def __call__(self, inputs):
    #     # print('--- ', __file__, ' inputs', inputs)
    #     return self.model(inputs)

    # def variables(self):
    #     return self.model.trainable_weights

    # def copy(self, scope=None):
    #     """copy network architecture"""
    #     scope = scope or self.scope + "_copy"
    #     with tf.variable_scope(scope):
    #         return NetworkKeras(
    #             state_shapes=self.state_shapes,
    #             action_size=self.action_size,
    #             nn_arch=self.nn_arch,
    #             num_atoms=self.num_atoms,
    #             v=self.v,
    #             have_action_input=self.has_action_input,
    #             use_next_state_input=self.use_next_state_input,
    #             scope=scope)
