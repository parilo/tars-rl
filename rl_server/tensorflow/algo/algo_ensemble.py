import tensorflow as tf


class AlgoEnsemble:
    
    def __init__(self, algorithms, placeholders):
        self._algos = algorithms
        
        self.actor_lr_ph = placeholders[0]
        self.critic_lr_ph = placeholders[1]
        self.states_ph = placeholders[2]
        self.actions_ph = placeholders[3]
        self.rewards_ph = placeholders[4]
        self.next_states_ph = placeholders[5]
        self.dones_ph = placeholders[6]

        self.actor_loss_op = [algo.get_policy_loss_op() for algo in self._algos]
        self.critic_loss_op = [algo.get_value_loss_op() for algo in self._algos]
        self.actor_update_op = [algo.get_actor_update_op() for algo in self._algos]
        self.critic_update_op = [algo.get_critic_update_op() for algo in self._algos]
            
        self.train_op = tf.tuple(
            self.critic_loss_op + 
            self.actor_loss_op +
            self.critic_update_op +
            self.actor_update_op
        )
        
        self.targets_init_op = tf.group(
            [algo.get_targets_init_op() for algo in self._algos])
        self.target_actor_update_op = tf.group(
            [algo.get_target_actor_update_op() for algo in self._algos])
        self.target_critic_update_op = tf.group(
            [algo.get_target_critic_update_op() for algo in self._algos])
    
    def act_batch(self, sess, states):
        raise NotImplementedError
        
    def act_batch_deterministic(self, sess, states):
        raise NotImplementedError
        
    def act_batch_with_gradients(self, sess, states):
        raise NotImplementedError

    def target_network_init(self, sess):
        sess.run(self.targets_init_op)
        
    def get_batch_size(self, step_index):
        return max([algo.get_batch_size(step_index) for algo in self._algos])
        
    def train(self, sess, step_index, batch, actor_update=True, critic_update=True):
        actor_lr = self._algos[0].get_actor_lr(step_index)
        critic_lr = self._algos[0].get_critic_lr(step_index)
        feed_dict = {
            self.actor_lr_ph: actor_lr,
            self.critic_lr_ph: critic_lr,
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops_ = sess.run(self.train_op, feed_dict=feed_dict)
        losses_values = ops_[:len(self._algos) * 2]
        return [actor_lr, critic_lr] + losses_values
    
    def get_td_errors(self, sess, batch):
        raise NotImplementedError
        
    def target_critic_update(self, sess):
        sess.run(self.target_critic_update_op)
        
    def target_actor_update(self, sess):
        sess.run(self.target_actor_update_op)

    def get_weights(self, sess, index=0):
        return self._algos[index].get_weights(sess)
