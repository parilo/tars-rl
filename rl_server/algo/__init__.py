import importlib

algo_names = {
    'tensorflow': [
        'ddpg',
        'categorical_ddpg',
        'quantile_ddpg',
        'td3',
        'quantile_td3',
        'sac',
        'env_learning',
        'dqn',
        'td_dqn',
        'sac_discrete'
    ],

    'torch': [
        'cross_entropy_method',
        'ddpg',
        'pddm_cem',
        'pddm_cem_lstm',
        'td3'
    ]
}

algo_create_funcs = {}

for framework, algo_names_list in algo_names.items():
    for algo_name in algo_names_list:
        algo_module = importlib.import_module('rl_server.' + framework + '.algo.' + algo_name)

        if framework not in algo_create_funcs:
            algo_create_funcs[framework] = {}

        algo_create_funcs[framework][algo_name] = getattr(algo_module, 'create_algo')
