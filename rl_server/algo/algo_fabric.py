import copy
import importlib

from rl_server.algo import algo_create_funcs


def combine_with_base_network(base_network_params, network_params):
    result_network_params = copy.deepcopy(network_params)
    result_network_params['nn_arch'] = base_network_params['nn_arch'] + network_params['nn_arch']
    return result_network_params


def get_network_params(algo_config, network_name):
    network_params = copy.deepcopy(algo_config.as_obj()[network_name])
    if 'base_network' in network_params:
        network_params = combine_with_base_network(
            copy.deepcopy(algo_config.as_obj()[network_params['base_network']]),
            network_params
        )
        del network_params['base_network']
    return network_params


def get_optimizer_class(optimizer_info):
    optim_module = importlib.import_module(optimizer_info['module'])
    return getattr(optim_module, optimizer_info['class'])


def create_algorithm(
    algo_config,
    placeholders=None,
    scope_postfix=0
):
    name = algo_config.algo_name
    print('--- creating {}'.format(name))

    if algo_config.framework == 'tensorflow':
        scope_postfix = str(scope_postfix)
        return algo_create_funcs[algo_config.framework][name](algo_config, placeholders, scope_postfix)

    elif algo_config.framework == 'torch':
        return algo_create_funcs[algo_config.framework][name](algo_config)
