import importlib

algo_names = [
    'ddpg',
    'categorical_ddpg',
    'quantile_ddpg',
    'td3',
    'quantile_td3',
    'sac',
    'env_learning'
]

algo_create_funcs = {}

for algo_name in algo_names:
    algo_module = importlib.import_module('rl_server.tensorflow.algo.' + algo_name)
    algo_create_funcs[algo_name] = getattr(algo_module, 'create_algo')
