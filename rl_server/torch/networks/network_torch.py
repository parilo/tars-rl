import importlib

import torch as t
import torch.nn as nn
import torch.distributions as dist
import numpy as np


def _calc_branch(branch, branch_input_values, random=True, hidden_state=None, hidden_state_output=None):
    x = branch_input_values
    for layer in branch['layers']:
        layer_args = {'x': x}
        if hasattr(layer, 'support_train_eval'):
            layer_args['random'] = random

        use_hidden_state = hasattr(layer, 'use_hidden_state') and layer.use_hidden_state is True
        if use_hidden_state:
            layer_args['hidden_state'] = hidden_state

        x = layer(**layer_args)

        if use_hidden_state:
            if hidden_state_output is not None:
                hidden_state_output.append(x[1])
            x = x[0]

    return x


def _calc_branch_with_dependencies(branches, branch_name, x, random=True, hidden_state=None, hidden_state_output=None):

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
            branch_input_values.append(_calc_branch_with_dependencies(
                branches,
                branch_input_name,
                x,
                random,
                hidden_state)
            )

    if len(branch_input_values) == 1:
        branch_input_values = branch_input_values[0]
    return _calc_branch(
        branches[branch_name],
        branch_input_values,
        random=random,
        hidden_state=hidden_state,
        hidden_state_output=hidden_state_output
    )


class LayerBase:

    def get_module(self):
        return None


class TorchLayer(LayerBase):

    def __init__(self, layer, args=None, expand_input_list=False, use_hidden_state=False):
        self._layer = layer
        self._args = args
        self._expand_input_list = expand_input_list
        self.use_hidden_state = use_hidden_state

    def __call__(self, x, hidden_state=None):
        if self._args is None:
            if self._expand_input_list:
                return self._layer(*x)
            else:

                if self.use_hidden_state:
                    output = self._layer(x, hidden_state)
                else:
                    output = self._layer(x)

                return output
        else:
            if self._expand_input_list:
                return self._layer(*x, **self._args)
            else:

                if self.use_hidden_state:
                    output = self._layer(x, hidden_state, **self._args)
                else:
                    output = self._layer(x, **self._args)

                return output

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


class ReshapeLayer(LayerBase):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.view(-1, *list(self.shape))


class SparseRandomNormal(LayerBase):

    def __init__(self, size, sigma, period):
        self.size = size
        self.sigma = sigma
        self.period = period
        self._call_index = 0
        self.support_train_eval = True
        # self._sin_t = 0

    # def __call__(self, x, train=True):
    #
    #     batch_size = x.shape[0]
    #     device = x.device
    #
    #     if train:
    #
    #         if self._call_index % self.period == 0 or x.shape[0] != self._phases.shape[0]:
    #             # redefine phases
    #             self._phases = 100 * t.rand((batch_size, self.size)).to(device)
    #             self._call_index = 0
    #
    #         sinarg = self._phases + t.tensor(self._sin_t / 100.0).to(device)
    #         ampl = t.tensor(self.sigma).to(device)
    #         output = ampl * t.sin(t.ones((batch_size, self.size)).to(device) * sinarg)
    #
    #         self._sin_t += 1
    #         self._call_index += 1
    #
    #     else:
    #         output = t.zeros((batch_size, self.size)).to(device)
    #
    #     return output

    def __call__(self, x, random=True):

        batch_size = x.shape[0]
        device = x.device

        if random:
            if self._call_index == 0 or x.shape != self.output.shape:

                distribution = dist.multivariate_normal.MultivariateNormal(
                    t.zeros((batch_size, self.size)).to(device),
                    covariance_matrix=self.sigma * t.eye(self.size).to(device)
                )

                self.output = distribution.sample()

            self._call_index += 1
            if self._call_index % self.period == 0:
                self._call_index = 0
        else:
            self.output = t.zeros((batch_size, self.size)).to(device)

        return self.output


def process_special_layers(layer_data):
    layer_type = layer_data['type']
    if layer_type == 'Flatten':
        return FlattenLayer()
    elif layer_type == 'Reshape':
        return ReshapeLayer(**layer_data['args'])
    elif layer_type == 'SparseRandomNormal':
        return SparseRandomNormal(**layer_data['args'])
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
                    layer = TorchFunc(layer_func, layer_args, layer_data.get('expand_input_list', False))

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

                    use_hidden_state = layer_data.get('use_hidden_state', False)
                    layer = TorchLayer(torch_layer_class_instance, use_hidden_state=use_hidden_state)

            layers.append(layer)

        return layers

    def build_model(self):

        branches = {}

        # if isinstance(self.nn_arch, list):
        assert isinstance(self.nn_arch, list), 'nn_arch must be list'

        for branch in self.nn_arch:

            branch_layers = self.process_layers(branch['layers'])
            branches[branch['name']] = {
                'layers': branch_layers,
                'input': branch['input']
            }

        # else:
        #     layers = self.process_layers(self.nn_arch['layers'], input_state[0])
        #     branches['output'] = {
        #         'layers': layers,
        #         'input': 0
        #     }

        for name, branch_info in branches.items():
            for i, layer in enumerate(branch_info['layers']):
                layer_torch_module = layer.get_module()
                if layer_torch_module is not None:
                    self.add_module('{}_layer_{}'.format(name, i), layer_torch_module)
                    # print('--- add_module', '{}_layer_{}'.format(name, i))

        return branches

    def forward(self, x, random=True, hidden_state=None):
        """
        :param x: list
        :return:
        """

        if 'output' not in self._branches:
            raise RuntimeError(
                'you must have branch with name "output" in nn_arch in the config'
                'or not to use branches'
            )

        hidden_state_output = []

        model_output = _calc_branch_with_dependencies(
            self._branches,
            'output',
            x,
            random=random,
            hidden_state=hidden_state,
            hidden_state_output=hidden_state_output
        )

        if hidden_state is None:
            return model_output
        else:
            return model_output, hidden_state_output

    def reset_states(self):
        pass

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