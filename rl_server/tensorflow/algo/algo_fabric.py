import tensorflow as tf
from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.quantile_td3 import QuantileTD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.algo.base_algo import create_placeholders


def create_algorithm(
    observation_shapes,
    state_shapes,
    action_size,
    algo_config,
    placeholders=None,
    scope_postfix=0
):
    if algo_config['use_lstm_networks']:
        from rl_server.tensorflow.networks.actor_networks_lstm import ActorNetwork
        from rl_server.tensorflow.networks.critic_networks_lstm import CriticNetwork
    else:
        from rl_server.tensorflow.networks.actor_networks import ActorNetwork, GaussActorNetwork
        from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork

    name = algo_config['algo_name']
    print('--- creating {}'.format(name))
    
    scope_postfix = str(scope_postfix)
    actor_scope = "actor_" + scope_postfix
    critic_scope = "critic_" + scope_postfix
    algo_scope = "algorithm_" + scope_postfix
    
    if placeholders is None:
        placeholders = create_placeholders(state_shapes, action_size)
    
    actor_lr = placeholders[0]
    critic_lr = placeholders[1]

    if name != "sac" and name != "td3" and name != "quantile_td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["actor"],
            scope=actor_scope)

        if name == "ddpg":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **algo_config["critic"],
                scope=critic_scope)
            DDPG_algorithm = DDPG
        elif name == "categorical":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **algo_config["critic"],
                scope=critic_scope)
            DDPG_algorithm = CategoricalDDPG
        elif name == "quantile":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **algo_config["critic"],
                # num_atoms=128,
                scope=critic_scope)
            DDPG_algorithm = QuantileDDPG
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
            **algo_config["algorithm"],
            scope=algo_scope,
            placeholders=placeholders)

    elif name == "sac":

        actor = GaussActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["actor"],
            scope=actor_scope)

        critic_v_params = dict(algo_config["critic"])
        critic_v_params['action_insert_block'] = -1
        critic_v = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **critic_v_params,
            scope="critic_v_" + scope_postfix)
        critic_q1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
            scope="critic_q1_" + scope_postfix)
        critic_q2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
            scope="critic_q2_" + scope_postfix)

        agent_algorithm = SAC(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
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
            **algo_config["algorithm"],
            scope=algo_scope,
            placeholders=placeholders)

    elif name == "td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["actor"],
            scope=actor_scope)
       
        critic1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
            scope="critic1_" + scope_postfix)
        critic2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
            scope="critic2_" + scope_postfix)

        agent_algorithm = TD3(
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
            **algo_config["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config["actor_optim"],
            critic_optim_schedule=algo_config["critic_optim"],
            training_schedule=algo_config["training"])
        
    elif name == "quantile_td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["actor"],
            scope=actor_scope)
       
        critic1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
            scope="critic1_" + scope_postfix)
        critic2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **algo_config["critic"],
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
            **algo_config["algorithm"],
            scope=algo_scope,
            placeholders=placeholders,
            actor_optim_schedule=algo_config["actor_optim"],
            critic_optim_schedule=algo_config["critic_optim"],
            training_schedule=algo_config["training"])
    
    return agent_algorithm
