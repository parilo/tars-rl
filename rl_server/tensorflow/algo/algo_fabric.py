import copy
import importlib

from rl_server.tensorflow.algo import algo_create_funcs


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
    # algorithm
    name = algo_config.algo_name
    print('--- creating {}'.format(name))
    scope_postfix = str(scope_postfix)
    return algo_create_funcs[name](algo_config, placeholders, scope_postfix)



    # if name == 'dqn':
    #
    #     critic_params = copy.deepcopy(algo_config.as_obj()["critic"])
    #
    #     critic = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **critic_params,
    #         scope=critic_scope,
    #         action_insert_block=-1
    #     )
    #
    #     from rl_server.tensorflow.algo.dqn import DQN
    #     agent_algorithm = DQN(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         critic=critic,
    #         critic_optimizer=tf.train.AdamOptimizer(
    #             learning_rate=critic_lr),
    #         **algo_config.as_obj()["algorithm"],
    #         scope=algo_scope,
    #         placeholders=placeholders,
    #         critic_optim_schedule=algo_config.as_obj()["critic_optim"],
    #         training_schedule=algo_config.as_obj()["training"])
    #
    # elif name == 'dqn_sac':
    #
    #     # base_network = copy.deepcopy(algo_config.as_obj()["base_network"])
    #     critic_q_params = copy.deepcopy(algo_config.as_obj()["critic_q"])
    #     critic_v_params = copy.deepcopy(algo_config.as_obj()["critic_v"])
    #     actor_params = copy.deepcopy(algo_config.as_obj()["actor"])
    #
    #     critic_q_params = combine_with_base_network(
    #         copy.deepcopy(algo_config.as_obj()[critic_q_params['base_network']]),
    #         critic_q_params
    #     )
    #     del critic_q_params['base_network']
    #
    #     critic_v_params = combine_with_base_network(
    #         copy.deepcopy(algo_config.as_obj()[critic_v_params['base_network']]),
    #         critic_v_params
    #     )
    #     del critic_v_params['base_network']
    #
    #     actor_params = combine_with_base_network(
    #         copy.deepcopy(algo_config.as_obj()[actor_params['base_network']]),
    #         actor_params
    #     )
    #     del actor_params['base_network']
    #
    #     critic_q = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **critic_q_params,
    #         scope=critic_scope,
    #         action_insert_block=-1
    #     )
    #
    #     critic_v = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **critic_v_params,
    #         scope=critic_scope,
    #         action_insert_block=-1
    #     )
    #
    #     actor = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **actor_params,
    #         scope=critic_scope,
    #         action_insert_block=-1
    #     )
    #
    #     actor_lr = tf.placeholder(tf.float32, (), "actor_lr")
    #
    #     from rl_server.tensorflow.algo.dqn_sac import DQN_SAC
    #     agent_algorithm = DQN_SAC(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         critic_q=critic_q,
    #         critic_v=critic_v,
    #         policy=actor,
    #         critic_optimizer=tf.train.AdamOptimizer(
    #             learning_rate=critic_lr),
    #         actor_optimizer=tf.train.AdamOptimizer(
    #             learning_rate=actor_lr),
    #         **algo_config.as_obj()["algorithm"],
    #         scope=algo_scope,
    #         placeholders=placeholders,
    #         actor_lr=actor_lr,
    #         critic_optim_schedule=algo_config.as_obj()["critic_optim"],
    #         actor_optim_schedule=algo_config.as_obj()["actor_optim"],
    #         training_schedule=algo_config.as_obj()["training"])
    #
    # elif name == 'dqn_td3':
    #
    #     critic_1 = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **critic_params,
    #         scope=critic_scope + '_1',
    #         action_insert_block=-1
    #     )
    #
    #     critic_2 = CriticNetwork(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         **critic_params,
    #         scope=critic_scope + '_2',
    #         action_insert_block=-1
    #     )
    #
    #     from rl_server.tensorflow.algo.dqn_td3 import DQN_TD3
    #     agent_algorithm = DQN_TD3(
    #         state_shapes=state_shapes,
    #         action_size=action_size,
    #         critic_1=critic_1,
    #         critic_2=critic_2,
    #         critic_optimizer=tf.train.AdamOptimizer(
    #             learning_rate=critic_lr),
    #         **algo_config.as_obj()["algorithm"],
    #         scope=algo_scope,
    #         placeholders=placeholders,
    #         critic_optim_schedule=algo_config.as_obj()["critic_optim"],
    #         training_schedule=algo_config.as_obj()["training"])
