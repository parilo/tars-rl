import importlib

import torch as t
import torch.nn as nn
import numpy as np


def _calc_branch(branch, branch_input_values):
    x = branch_input_values
    # print('--- _calc_branch', x)
    for layer in branch['layers']:
        x = layer(x)
    return x


def _calc_branch_with_dependencies(branches, branch_name, x):
    # print('--- _calc_branch_with_dependencies', branch_name)

    branch_input = branches[branch_name]['input']
    if not isinstance(branch_input, list):
        branch_input = [branch_input]

    branch_input_values = []
    for branch_input_name in branch_input:
        if isinstance(branch_input_name, int):
            # print('--- input branch is int')
            branch_input_values.append(x[branch_input_name])
        elif branch_input_name == 'action':
            branch_input_values.append(x[-1])
        else:
            # print('--- input branch is not int')
            branch_input_values.append(_calc_branch_with_dependencies(branches, branch_input_name, x))

    if len(branch_input_values) == 1:
        branch_input_values = branch_input_values[0]
    return _calc_branch(branches[branch_name], branch_input_values)


class LayerBase:

    def get_module(self):
        return None


class TorchLayer(LayerBase):

    def __init__(self, layer, args=None):
        self._layer = layer
        self._args = args

    def __call__(self, x):
        if self._args is None:
            return self._layer(x)
        else:
            return self._layer(x, **self._args)

    def get_module(self):
        return self._layer


class TorchFunc(TorchLayer):

    def get_module(self):
        return None


class FlattenLayer(LayerBase):

    def __call__(self, x):
        if len(x.shape) > 1:
            return x.view((x.shape[0], np.product(x.shape[1:])))
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
        device='cpu'
        # v=(-10., 10.),
        # scope=None
    ):
        super().__init__()
        self.nn_arch = nn_arch
        self.has_action_input = have_action_input
        self.use_next_state_input = use_next_state_input
        self.num_atoms = num_atoms
        self.device = device
        # self.v = v
        # self.scope = scope or "CriticNetwork"
        self._branches = self.build_model()

    def _apply_initializer(self, initializer_data, torch_layer_class_instance):
        tensor = getattr(torch_layer_class_instance, initializer_data['tensor'])
        init_module = importlib.import_module(initializer_data['module'])
        init_func = getattr(init_module, initializer_data['func'])
        if 'args' in initializer_data:
            init_func(tensor, **initializer_data['args'])
        else:
            init_func(tensor)

    def process_layers(self, layers_info):

        # print('--- branch')
        nn_module = importlib.import_module('torch.nn')

        layers = []

        for layer_i, layer_data in enumerate(layers_info):

            layer = process_special_layers(layer_data)

            if layer is None:

                # module which contains layer
                if 'module' in layer_data:
                    layer_module = importlib.import_module(layer_data['module'])
                else:
                    layer_module = nn_module

                layer_is_func = layer_data.get('is_func', False)
                if layer_is_func:
                    # layer is function
                    layer_func = getattr(layer_module, layer_data['type'])
                    layer_args = layer_data.get('args', None)
                    layer = TorchFunc(layer_func, layer_args)

                else:
                    # layer is class
                    LayerClass = getattr(layer_module, layer_data['type'])
                    if 'args' in layer_data:
                        # process_layer_args(layer_data['args'])
                        torch_layer_class_instance = LayerClass(**layer_data['args'])
                    else:
                        torch_layer_class_instance = LayerClass()

                    if 'initializers' in layer_data:
                        for initializer_data in layer_data['initializers']:
                            self._apply_initializer(initializer_data, torch_layer_class_instance)

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
                layer_torch_module = layer.get_module()
                if layer_torch_module is not None:
                    self.add_module('{}_layer_{}'.format(name, i), layer_torch_module)
                    # print('--- add_module', '{}_layer_{}'.format(name, i))

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

    def get_weights(self):
        weights = {}
        for name, value in self.state_dict().items():
            weights[name] = value.cpu().detach().numpy()
        return weights

    def set_weights(self, weights):
        state_dict = {}
        for name, value in weights.items():
            state_dict[name] = t.tensor(value).to(self.device)
        self.load_state_dict(state_dict)

    def copy(self):
        """copy network"""
        new_model = NetworkTorch(
            nn_arch=self.nn_arch,
            have_action_input=self.has_action_input,
            use_next_state_input=self.use_next_state_input,
            num_atoms=self.num_atoms,
            device=self.device
        )

        new_model.load_state_dict(self.state_dict())
        return new_model