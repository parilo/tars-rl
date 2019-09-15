import tensorflow as tf

from .base_algo import BaseAlgo, network_update, target_network_update
from rl_server.tensorflow.algo.model_weights_tool import ModelWeightsTool
from rl_server.algo.algo_fabric import get_network_params, get_optimizer_class
from rl_server.tensorflow.networks.network_keras import NetworkKeras
from rl_server.tensorflow.algo.base_algo import create_placeholders


def create_placeholders_el(state_shapes, action_size, scope="placeholders"):
    ph = create_placeholders(state_shapes, action_size, scope)
    return ph[1:]

def create_algo(algo_config, placeholders, scope_postfix):

    _, _, state_shapes, action_size = algo_config.get_env_shapes()
    if placeholders is None:
        placeholders = create_placeholders_el(state_shapes, action_size)

    algo_scope = 'el_' + scope_postfix
    critic_lr = placeholders[0]
    critic_optim_info = algo_config.as_obj()['critic_optim']

    env_model_params = get_network_params(algo_config, "env_model")
    reward_model_params = get_network_params(algo_config, "reward_model")
    done_model_params = get_network_params(algo_config, "done_model")
    actor_params = get_network_params(algo_config, "actor")

    env_model = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **env_model_params,
        scope='env_model_' + scope_postfix)

    reward_model = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **reward_model_params,
        scope='reward_model_' + scope_postfix)

    done_model = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **done_model_params,
        scope='done_model_' + scope_postfix)

    actor = NetworkKeras(
        state_shapes=state_shapes,
        action_size=action_size,
        **actor_params,
        scope='actor_' + scope_postfix)

    return EnvLearning(
        state_shapes=state_shapes,
        action_size=action_size,
        actor=actor,
        env_model=env_model,
        reward_model=reward_model,
        done_model=done_model,
        optimizer=tf.train.AdamOptimizer(learning_rate=critic_lr),
        n_step=algo_config.as_obj()["algorithm"]['n_step'],
        scope="algorithm",
        placeholders=placeholders,
        optim_schedule=algo_config.as_obj()["critic_optim"],
        training_schedule=algo_config.as_obj()["training"]
    )


class EnvLearning:
    def __init__(
        self,
        state_shapes,
        action_size,
        actor,
        env_model,
        reward_model,
        done_model,
        optimizer,
        n_step=1,
        scope="algorithm",
        placeholders=None,
        optim_schedule={'schedule': [{'limit': 0, 'lr': 1e-4}]},
        training_schedule={'schedule': [{'limit': 0, 'batch_size_mult': 1}]}
    ):
        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._env_model = env_model
        self._reward_model = reward_model
        self._done_model = done_model
        self._optimizer = optimizer
        self._n_step = n_step
        self._scope = scope
        self._placeholders = placeholders
        self._optim_schedule = optim_schedule
        self._training_schedule = training_schedule

        self._actor_weights_tool = ModelWeightsTool(actor)
        self._env_model_weights_tool = ModelWeightsTool(env_model)
        self._reward_model_weights_tool = ModelWeightsTool(reward_model)
        self._done_model_weights_tool = ModelWeightsTool(done_model)

        with tf.name_scope(scope):
            self.build_graph()

    def compose_next_state(self, prev_state, next_obs):
        return [tf.concat([
            prev_state[0][:, 1:, :],
            tf.expand_dims(next_obs, axis=-2)
        ], axis=-2)]

    def get_step_mask(self, pred_dones):
        max_ind = tf.argmax(pred_dones, axis=-1)
        max_val = tf.reduce_max(pred_dones, axis=-1)
        done_ind = tf.where(max_val > 0.5, x=max_ind, y=self._n_step * tf.ones_like(max_ind))
        dones_mask = tf.cast(tf.sequence_mask(done_ind + 1, self._n_step), dtype=tf.float32)
        return dones_mask

    def build_graph(self):
        self.critic_lr_ph = self._placeholders[0]
        self.states_ph = self._placeholders[1]
        self.actions_ph = self._placeholders[2]
        self.rewards_ph = self._placeholders[3]
        self.next_states_ph = self._placeholders[4]
        self.dones_ph = self._placeholders[5]

        with tf.name_scope("taking_action"):
            self._actions = self._actor(self.states_ph)

        with tf.name_scope("env_model_update"):
            self._predicted_next_state = self._env_model(self.states_ph + [self.actions_ph])
            print('--- self.next_states_ph', self.next_states_ph)
            print('--- self._predicted_next_state', self._predicted_next_state)
            # predicting and comparing only last observation
            self._env_model_loss = tf.losses.huber_loss(self._predicted_next_state, self.next_states_ph[0][:, -1, :])
            self._env_model_update = network_update(
                self._env_model_loss,
                self._env_model,
                self._optimizer
            )

        with tf.name_scope("reward_model_update"):
            self._predicted_reward = self._reward_model(self.states_ph + self.next_states_ph)[:, -1]
            print('--- self._predicted_reward', self._predicted_reward)
            self._reward_model_loss = tf.losses.huber_loss(self._predicted_reward, self.rewards_ph)
            self._reward_model_update = network_update(
                self._reward_model_loss,
                self._reward_model,
                self._optimizer
            )

        with tf.name_scope("done_model_update"):
            self._predicted_done = self._done_model(self.states_ph + self.next_states_ph + [self.actions_ph])[:, -1]
            print('--- self._predicted_done', self._predicted_done)
            self._done_model_loss = tf.losses.huber_loss(self._predicted_done, self.dones_ph)
            self._done_model_update = network_update(
                self._done_model_loss,
                self._done_model,
                self._optimizer
            )

        with tf.name_scope("actor_update"):
            pred_rewards = []
            pred_done = []
            p_state = self.states_ph
            print('--- n step', self._n_step)
            for _ in range(self._n_step):
                p_action = self._actor(p_state)
                p_next_state = self._env_model(p_state + [p_action])
                p_next_state = self.compose_next_state(p_state, p_next_state)
                print('--- p_next_state', p_next_state)
                p_reward = self._reward_model(p_state + p_next_state)[:, -1]
                p_done = self._done_model(p_state + p_next_state + [p_action])[:, -1]
                pred_rewards.append(p_reward)
                pred_done.append(p_done)
                p_state = p_next_state

            all_dones = tf.stack(pred_done, axis=-1)
            all_dones_mask = self.get_step_mask(all_dones)
            self._all_dones_mask = all_dones_mask
            print('--- all_dones', all_dones)
            print('--- all_dones_mask', all_dones_mask)

            all_steps_rewards = tf.stack(pred_rewards, axis=-1)
            all_steps_rewards = tf.multiply(all_steps_rewards, all_dones_mask)
            all_steps_rewards = tf.reduce_sum(all_steps_rewards, axis=-1)
            self._actor_loss = -tf.reduce_mean(all_steps_rewards)
            print('--- all_steps_rewards', all_steps_rewards)

            self._actor_update = network_update(
                self._actor_loss,
                self._actor,
                self._optimizer
            )

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def get_schedule_params(self, schedule, step_index):
        for training_params in schedule['schedule']:
            if step_index >= training_params['limit']:
                return training_params
        return schedule['schedule'][0]

    def get_batch_size(self, step_index):
        return self.get_schedule_params(self._training_schedule, step_index)['batch_size']

    def get_critic_lr(self, step_index):
        return self.get_schedule_params(self._optim_schedule, step_index)['lr']

    def target_network_init(self, sess):
        # sess.run(self._targets_init_op)
        pass

    def act_batch(self, sess, states):
        feed_dict = dict(zip(self.states_ph, states))
        actions = sess.run(self._actions, feed_dict=feed_dict)
        return actions.tolist()

    def train(self, sess, step_index, batch, actor_update=True, critic_update=True):
        # if step_index % 1 == 0:
        return self.train_2(sess, step_index, batch)
        # else:
        #     return self.train_1(sess, step_index, batch)

    def train_1(self, sess, step_index, batch, actor_update=True, critic_update=True):
        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [
            self._env_model_loss,
            self._reward_model_loss,
            self._done_model_loss,
            self._env_model_update,
            self._reward_model_update,
            self._done_model_update
        ]
        ops_ = sess.run(ops, feed_dict=feed_dict)

        return {
            'critic lr':  critic_lr,
            'e loss': ops_[0],
            'r loss': ops_[1],
            'd loss': ops_[2]
        }

    def train_2(self, sess, step_index, batch, actor_update=True, critic_update=True):
        critic_lr = self.get_critic_lr(step_index)
        feed_dict = {
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops = [
            self._env_model_loss,
            self._reward_model_loss,
            self._done_model_loss,
            self._actor_loss,
            self._env_model_update,
            self._reward_model_update,
            self._done_model_update,
            self._actor_update
        ]
        ops_ = sess.run(ops, feed_dict=feed_dict)

        return {
            'critic lr':  critic_lr,
            'e loss': ops_[0],
            'r loss': ops_[1],
            'd loss': ops_[2],
            'a loss': ops_[3]
        }

    def target_actor_update(self, sess):
        pass

    def target_critic_update(self, sess):
        pass

    def get_weights(self, sess, index=0):
        return {
            'actor': self._actor_weights_tool.get_weights(sess),
            'env_model': self._env_model_weights_tool.get_weights(sess),
            'reward_model': self._reward_model_weights_tool.get_weights(sess),
            'done_model': self._done_model_weights_tool.get_weights(sess)
        }

    def set_weights(self, sess, weights):
        self._actor_weights_tool.set_weights(sess, weights['actor'])
        self._env_model_weights_tool.set_weights(sess, weights['env_model'])
        self._reward_model_weights_tool.set_weights(sess, weights['reward_model'])
        self._done_model_weights_tool.set_weights(sess, weights['done_model'])

    def reset_states(self):
        self._actor.reset_states()
        self._env_model.reset_states()
        self._reward_model.reset_states()
        self._done_model.reset_states()
