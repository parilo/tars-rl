from rl_server.tensorflow.algo.ddpg import DDPG
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.algo.quantile_ddpg import QuantileDDPG
from rl_server.tensorflow.algo.td3 import TD3
from rl_server.tensorflow.algo.sac import SAC
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork

def create_algorithm(name, hparams):

    history_len = hparams["server"]["history_length"]
    state_shapes = [(history_len, hparams["env"]["obs_size"],)]
    action_size = hparams["env"]["action_size"]

    if name != "sac" and name != "td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            scope="actor")

        if name == "ddpg":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                scope="critic")
            DDPG_algorithm = DDPG
        elif name == "categorical":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                # num_atoms=101,
                # v=(-100., 900.),
                scope="critic")
            DDPG_algorithm = CategoricalDDPG
        elif name == "quantile":
            critic = CriticNetwork(
                state_shape=state_shapes[0],
                action_size=action_size,
                **hparams["critic"],
                num_atoms=128,
                scope="critic")
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
            **hparams["algorithm"])

    elif name == "sac":

        actor = GMMActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            num_components=4,
            scope="actor")

        critic_v = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            action_insert_block=-1,
            scope="critic_v")
        critic_q = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic_q")

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
            reward_scale=200)

    elif name == "td3":

        actor = ActorNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["actor"],
            scope="actor")
       
        critic1 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic1")
        critic2 = CriticNetwork(
            state_shape=state_shapes[0],
            action_size=action_size,
            **hparams["critic"],
            scope="critic2")

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
            **hparams["algorithm"])
    
    return agent_algorithm
