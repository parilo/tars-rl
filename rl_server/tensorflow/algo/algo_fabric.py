from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork

def create_algorithm(name, hparams, placeholders=None, scope_postfix=0):

    history_len = hparams["server"]["history_length"]
    state_shapes = [(history_len, hparams["env"]["obs_size"],)]
    action_size = hparams["env"]["action_size"]
    scope_postfix = str(scope_postfix)
    actor_scope = "actor_" + scope_postfix
    critic_scope = "critic_" + scope_postfix
    algo_scope = "algorithm_" + scope_postfix

    if name != "sac" and name != "td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            scope=actor_scope)

        if name == "ddpg":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                scope=critic_scope)
            DDPG_algorithm = DDPG
        elif name == "categorical":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                scope=critic_scope)
            DDPG_algorithm = CategoricalDDPG
        elif name == "quantile":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                num_atoms=128,
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
                learning_rate=hparams["actor_optim"]["lr"]),
            critic_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["critic_optim"]["lr"]),
            **hparams["algorithm"],
            scope=algo_scope,
            placeholders=placeholders)

    elif name == "sac":

        actor = GMMActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            num_components=4,
            scope=actor_scope)

        critic_v = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            action_insert_block=-1,
            scope="critic_v_" + scope_postfix)
        critic_q = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic_q_" + scope_postfix)

        agent_algorithm = SAC(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            critic_v=critic_v,
            critic_q=critic_q,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["actor_optim"]["lr"]),
            critic_v_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["critic_optim"]["lr"]),
            critic_q_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["critic_optim"]["lr"]),
            **hparams["algorithm"],
            reward_scale=200,
            scope=algo_scope,
            placeholders=placeholders)

    elif name == "td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            scope=actor_scope)
       
        critic1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic1_" + scope_postfix)
        critic2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic2_" + scope_postfix)

        agent_algorithm = TD3(
            state_shapes=state_shapes,
            action_size=action_size,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["actor_optim"]["lr"]),
            critic1_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["critic_optim"]["lr"]),
            critic2_optimizer=tf.train.AdamOptimizer(
                learning_rate=hparams["critic_optim"]["lr"]),
            **hparams["algorithm"],
            scope=algo_scope,
            placeholders=placeholders)
    
    return agent_algorithm
