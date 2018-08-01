import tensorflow as tf
from .ddpg import BaseDDPG


class TD3(BaseDDPG):
    def __init__(self,
                 state_shapes,
                 action_size,
                 actor,
                 critic1,
                 critic2,
                 actor_optimizer,
                 critic1_optimizer,
                 critic2_optimizer,
                 n_step=1,
                 gradient_clip=1.0,
                 action_noise_std=0.2,
                 action_noise_clip=0.5,
                 discount_factor=0.99,
                 target_actor_update_rate=1.0,
                 target_critic1_update_rate=1.0,
                 target_critic2_update_rate=1.0):

        self._state_shapes = state_shapes
        self._action_size = action_size
        self._actor = actor
        self._critic1 = critic1
        self._critic2 = critic2
        self._target_actor = actor.copy(scope='target_actor')
        self._target_critic1 = critic1.copy(scope='target_critic1')
        self._target_critic2 = critic2.copy(scope='target_critic2')
        self._actor_optimizer = actor_optimizer
        self._critic1_optimizer = critic1_optimizer
        self._critic2_optimizer = critic2_optimizer
        self._n_step = n_step
        self._grad_clip = gradient_clip
        self._act_noise_std = action_noise_std
        self._act_noise_clip = action_noise_clip
        self._gamma = discount_factor
        self._update_rates = [target_actor_update_rate, target_critic1_update_rate, target_critic2_update_rate]
        self._target_actor_update_rate = tf.constant(target_actor_update_rate)
        self._target_critic1_update_rate = tf.constant(target_critic1_update_rate)
        self._target_critic2_update_rate = tf.constant(target_critic2_update_rate)
        self._action_var = tf.Variable(tf.zeros((1, self._action_size)), dtype=tf.float32)

        self._create_placeholders()
        self._create_variables()

    def _get_q_values(self, states, actions):
        return self._critic1([states, actions])

    def _get_critic_update(self):

        # left hand side of the Bellman equation for both critics
        next_action = self._target_actor(self._next_state)
        act_noise = tf.random_normal(tf.shape(next_action), mean=0.0, stddev=self._act_noise_std)
        clip_noise = tf.clip_by_value(act_noise, -self._act_noise_clip, self._act_noise_clip)
        next_action = next_action + clip_noise

        next_q1 = self._target_critic1([self._next_state, next_action])
        next_q2 = self._target_critic2([self._next_state, next_action])
        next_q = tf.minimum(next_q1, next_q2)
        discount = self._gamma ** self._n_step
        target_q = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_q

        # left hand side of the Bellman equation
        agent_q1 = self._critic1([self._state, self._given_action])
        agent_q2 = self._critic2([self._state, self._given_action])

        # critic gradients and update rule
        critic1_loss = tf.losses.huber_loss(agent_q1, tf.stop_gradient(target_q))
        critic1_gradients = self._critic1_optimizer.compute_gradients(
            critic1_loss, var_list=self._critic1.variables())
        critic1_update = self._critic1_optimizer.apply_gradients(critic1_gradients)
        
        critic2_loss = tf.losses.huber_loss(agent_q2, tf.stop_gradient(target_q))
        critic2_gradients = self._critic2_optimizer.compute_gradients(
            critic2_loss, var_list=self._critic2.variables())
        critic2_update = self._critic2_optimizer.apply_gradients(critic2_gradients)

        return [critic1_loss, critic2_loss], tf.group(critic1_update, critic2_update)

    def _get_actor_update(self):

        # actor gradient and update rule
        actor_loss = -tf.reduce_mean(self._critic1([self._state, self._actor(self._state)]))
        actor_gradients = self._actor_optimizer.compute_gradients(
            actor_loss, var_list=self._actor.variables())

        actor_gradients_clip = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var)
                                for grad, var in actor_gradients]
        actor_update = self._actor_optimizer.apply_gradients(actor_gradients)
        return actor_loss, actor_update

    def _get_target_critic_update(self):
        target_critic1_update = BaseDDPG._update_target_network(
            self._critic1, self._target_critic1, self._target_critic1_update_rate)
        target_critic2_update = BaseDDPG._update_target_network(
            self._critic2, self._target_critic2, self._target_critic2_update_rate)
        return tf.group(target_critic1_update, target_critic2_update)

    def _get_targets_init(self):
        target_actor_update = BaseDDPG._update_target_network(self._actor, self._target_actor, 1.0)
        target_critic1_update = BaseDDPG._update_target_network(self._critic1, self._target_critic1, 1.0)
        target_critic2_update = BaseDDPG._update_target_network(self._critic2, self._target_critic2, 1.0)
        update_targets = tf.group(target_actor_update, target_critic1_update, target_critic2_update)
        return update_targets

    def _get_info(self):
        info = {}
        info['algo'] = 'td3'
        info['actor'] = self._actor.get_info()
        info['critic1'] = self._critic1.get_info()
        info['critic2'] = self._critic2.get_info()
        info['grad_clip'] = self._grad_clip
        info['action_noise_std'] = self._act_noise_std
        info['action_noise_clip'] = self._act_noise_clip
        info['discount_factor'] = self._gamma
        info['target_actor_update_rate'] = self._update_rates[0]
        info['target_critic1_update_rate'] = self._update_rates[1]
        info['target_critic2_update_rate'] = self._update_rates[2]
        return info
