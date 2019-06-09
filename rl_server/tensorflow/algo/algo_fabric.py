import copy

import tensorflow as tf

from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.quantile_td3 import QuantileTD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.algo.prioritized_ddpg import PrioritizedDDPG
from rl_server.tensorflow.algo.env_learning import EnvLearning
# from rl_server.tensorflow.networks.actor_networks_lstm import ActorNetwork as ActorNetworkLSTM
# from rl_server.tensorflow.networks.critic_networks_lstm import CriticNetwork as CriticNetworkLSTM
# from rl_server.tensorflow.networks.actor_networks import (
#     ActorNetwork as ActorNetworkFF,
#     GaussActorNetwork as GaussActorNetworkFF
# )
# from rl_server.tensorflow.networks.critic_networks import CriticNetwork as CriticNetworkFF
from rl_server.tensorflow.networks.network_keras import NetworkKeras


def combine_with_base_network(base_network_params, network_params):
    result_network_params = copy.deepcopy(network_params)
    result_network_params['nn_arch'] = base_network_params['nn_arch'] + network_params['nn_arch']
    return result_network_params


def get_network_params(algo_config, network_name):
    network_params = copy.deepcopy(algo_config.as_obj()[network_name])
    network_params = combine_with_base_network(
        copy.deepcopy(algo_config.as_obj()[network_params['base_network']]),
        network_params
    )
    del network_params['base_network']
    return network_params


def create_algorithm(
    algo_config,
    placeholders=None,
    scope_postfix=0
):
    # networks
    # if algo_config.isset('nn_engine'):
    # if algo_config.nn_engine == 'keras':

        # if algo_config.isset('actor'):
        #     from rl_server.tensorflow.networks.actor_networks_keras import ActorNetwork as ActorNetworkKeras
        #     ActorNetwork = ActorNetworkKeras
        #     actor_params = copy.deepcopy(algo_config.as_obj()["actor"])
        #     del actor_params['nn_engine']

        # from rl_server.tensorflow.networks.critic_networks_keras import CriticNetwork as CriticNetworkKeras
        # CriticNetwork = CriticNetworkKeras

        # if algo_config.isset('critic'):
        #     critic_params = copy.deepcopy(algo_config.as_obj()["critic"])

    # else:
    #     raise NotImplementedError()
    # else:
    #     if algo_config.isset('actor'):
    #         if algo_config.actor.lstm_network:
    #             ActorNetwork = ActorNetworkLSTM
    #             GaussActorNetwork = None
    #         else:
    #             ActorNetwork = ActorNetworkFF
    #             GaussActorNetwork = GaussActorNetworkFF
    #
    #         actor_params = copy.deepcopy(algo_config.as_obj()["actor"])
    #         del actor_params['lstm_network']
    #
    #     if algo_config.critic.lstm_network:
    #         CriticNetwork = CriticNetworkLSTM
    #     else:
    #         CriticNetwork = CriticNetworkFF
    #
    #     critic_params = copy.deepcopy(algo_config.as_obj()["critic"])
    #     del critic_params['lstm_network']

    # algorithm
    name = algo_config.algo_name
    print('--- creating {}'.format(name))
    
    scope_postfix = str(scope_postfix)
    actor_scope = "actor_" + scope_postfix
    critic_scope = "critic_" + scope_postfix
    algo_scope = "algorithm_" + scope_postfix

    _, _, state_shapes, action_size = algo_config.get_env_shapes()
    if placeholders is None:
        if name in ['dqn', 'dqn_td3', 'dqn_sac']:
            from rl_server.tensorflow.algo.base_algo_discrete import create_placeholders
            placeholders = create_placeholders(state_shapes)
            critic_lr = placeholders[0]
        else:
            from rl_server.tensorflow.algo.base_algo import create_placeholders
            placeholders = create_placeholders(state_shapes, action_size)
            actor_lr = placeholders[0]
            critic_lr = placeholders[1]

    if name == 'dqn':

        critic = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_params,
            scope=critic_scope,
            action_insert_block=-1
        )

        from rl_server.tensorflow.algo.dqn import DQN
        agent_algorithm = DQN(
            state_shapes=state_shapes,
            action_size=action_size,
            critic=critic,
            critic_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])

    elif name == 'dqn_sac':

        # base_network = copy.deepcopy(algo_config.as_obj()["base_network"])
        critic_q_params = copy.deepcopy(algo_config.as_obj()["critic_q"])
        critic_v_params = copy.deepcopy(algo_config.as_obj()["critic_v"])
        policy_params = copy.deepcopy(algo_config.as_obj()["policy"])

        critic_q_params = combine_with_base_network(
            copy.deepcopy(algo_config.as_obj()[critic_q_params['base_network']]),
            critic_q_params
        )
        del critic_q_params['base_network']

        critic_v_params = combine_with_base_network(
            copy.deepcopy(algo_config.as_obj()[critic_v_params['base_network']]),
            critic_v_params
        )
        del critic_v_params['base_network']

        policy_params = combine_with_base_network(
            copy.deepcopy(algo_config.as_obj()[policy_params['base_network']]),
            policy_params
        )
        del policy_params['base_network']

        critic_q = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_q_params,
            scope=critic_scope,
            action_insert_block=-1
        )

        critic_v = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_v_params,
            scope=critic_scope,
            action_insert_block=-1
        )

        policy = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **policy_params,
            scope=critic_scope,
            action_insert_block=-1
        )

        from rl_server.tensorflow.algo.dqn_sac import DQN_SAC
        agent_algorithm = DQN_SAC(
            state_shapes=state_shapes,
            action_size=action_size,
            critic_q=critic_q,
            critic_v=critic_v,
            policy=policy,
            critic_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])

    elif name == 'dqn_td3':

        critic_1 = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_params,
            scope=critic_scope + '_1',
            action_insert_block=-1
        )

        critic_2 = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_params,
            scope=critic_scope + '_2',
            action_insert_block=-1
        )

        from rl_server.tensorflow.algo.dqn_td3 import DQN_TD3
        agent_algorithm = DQN_TD3(
            state_shapes=state_shapes,
            action_size=action_size,
            critic_1=critic_1,
            critic_2=critic_2,
            critic_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])

    elif name == 'env_learning':

        env_model_params = copy.deepcopy(algo_config.as_obj()["env_model"])
        reward_model_params = copy.deepcopy(algo_config.as_obj()["reward_model"])
        done_model_params = copy.deepcopy(algo_config.as_obj()["done_model"])
        actor_params = copy.deepcopy(algo_config.as_obj()["actor"])

        env_model = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **env_model_params,
            scope='env_model_' + scope_postfix)

        reward_model = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **reward_model_params,
            scope='reward_model_' + scope_postfix)

        done_model = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **done_model_params,
            scope='done_model_' + scope_postfix)

        actor = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **actor_params,
            scope='actor_' + scope_postfix)

        agent_algorithm = EnvLearning(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            env_model=env_model,
            reward_model=reward_model,
            done_model=done_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=critic_lr),
            n_step=algo_config.as_obj()["algorithm"]['n_step'],
            scope="algorithm",
            placeholders=placeholders[1:],
            optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"]
        )

    elif name in ["ddpg", "categorical_ddpg", "quantile_ddpg"]:

        actor_params = copy.deepcopy(algo_config.as_obj()["actor"])

        actor = CriticNetwork(
            state_shapes=state_shapes,
            action_size=action_size,
            **actor_params,
            scope=actor_scope)

        if name == "ddpg":
            critic_params = copy.deepcopy(algo_config.as_obj()["critic"])
            critic = CriticNetwork(
                state_shapes=state_shapes,
                action_size=action_size,
                **critic_params,
                scope=critic_scope)

            if algo_config.server.use_prioritized_buffer:
                DDPG_algorithm = PrioritizedDDPG
            else:
                DDPG_algorithm = DDPG
        # elif name == "categorical_ddpg":
        #     assert not algo_config.server.use_prioritized_buffer, '{} have no prioritized version. use ddpg'.format(
        #         name)
        #     critic = CriticNetwork(
        #         state_shape=state_shapes[0],
        #         action_size=action_size,
        #         **critic_params,
        #         scope=critic_scope)
        #     DDPG_algorithm = CategoricalDDPG
        # elif name == "quantile_ddpg":
        #     assert not algo_config.server.use_prioritized_buffer, '{} have no prioritized version. use ddpg'.format(
        #         name)
        #     critic = CriticNetwork(
        #         state_shape=state_shapes[0],
        #         action_size=action_size,
        #         **critic_params,
        #         scope=critic_scope)
        #     DDPG_algorithm = QuantileDDPG
        else:
            raise NotImplementedError

        agent_algorithm = DDPG_algorithm(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            critic=critic,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=actor_lr),
            critic_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config.as_obj()["actor_optim"],
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])

    elif name == "sac":

        assert not algo_config.server.use_prioritized_buffer, '{} have no prioritized version. use ddpg'.format(name)

        policy_params = get_network_params(algo_config, 'policy')
        policy = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **policy_params,
            scope="policy_" + scope_postfix
        )

        critic_v_params = get_network_params(algo_config, 'critic_v')
        critic_v = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_v_params,
            scope="critic_v_" + scope_postfix
        )

        critic_q_params = get_network_params(algo_config, 'critic_q')
        critic_q1 = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_q_params,
            scope="critic_q1_" + scope_postfix
        )
        critic_q2 = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_q_params,
            scope="critic_q2_" + scope_postfix
        )

        agent_algorithm = SAC(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=policy,
            critic_v=critic_v,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=actor_lr),
            critic_v_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            critic_q1_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            critic_q2_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config.as_obj()["actor_optim"],
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])

    elif name == "td3":

        policy_params = get_network_params(algo_config, 'policy')
        policy = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **policy_params,
            scope="policy_" + scope_postfix
        )

        critic_1_params = get_network_params(algo_config, 'critic_1')
        critic_1 = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_1_params,
            scope="critic_1_" + scope_postfix
        )

        critic_2_params = get_network_params(algo_config, 'critic_2')
        critic_2 = NetworkKeras(
            state_shapes=state_shapes,
            action_size=action_size,
            **critic_2_params,
            scope="critic_2_" + scope_postfix
        )

        agent_algorithm = TD3(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=policy,
            critic1=critic_1,
            critic2=critic_2,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=actor_lr),
            critic1_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            critic2_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config.as_obj()["actor_optim"],
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])
        
    elif name == "quantile_td3":

        assert not algo_config.server.use_prioritized_buffer, '{} have no prioritized version. use ddpg'.format(name)

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **actor_params,
            scope=actor_scope)
       
        critic1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **critic_params,
            scope="critic1_" + scope_postfix)
        critic2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **critic_params,
            scope="critic2_" + scope_postfix)

        agent_algorithm = QuantileTD3(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=actor_lr),
            critic1_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            critic2_optimizer=tf.train.AdamOptimizer(
                learning_rate=critic_lr),
            **algo_config.as_obj()["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config.as_obj()["actor_optim"],
            critic_optim_schedule=algo_config.as_obj()["critic_optim"],
            training_schedule=algo_config.as_obj()["training"])
    
    return agent_algorithm
