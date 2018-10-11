import tensorflow as tf


class AlgoEnsemble:
    
    def __init__(self, algorithms, placeholders):
        self._algos = algorithms
        
        self.states_ph = placeholders[0]
        self.actions_ph = placeholders[1]
        self.rewards_ph = placeholders[2]
        self.next_states_ph = placeholders[3]
        self.dones_ph = placeholders[4]

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
        
    def train(self, sess, batch, actor_update=True, critic_update=True):
        feed_dict = {
            **dict(zip(self.states_ph, batch.s)),
            **{self.actions_ph: batch.a},
            **{self.rewards_ph: batch.r},
            **dict(zip(self.next_states_ph, batch.s_)),
            **{self.dones_ph: batch.done}}
        ops_ = sess.run(self.train_op, feed_dict=feed_dict)
        losses_values = ops_[:len(self._algos) * 2]
        return losses_values
    
    def get_td_errors(self, sess, batch):
        raise NotImplementedError
        
    def target_critic_update(self, sess):
        sess.run(self.target_critic_update_op)
        
    def target_actor_update(self, sess):
        sess.run(self.target_actor_update_op)

    def get_weights(self, sess, index=0):
        return self._algos[index].get_weights(sess)

    def _get_info(self):
        info = {}
        info["algo"] = "ensemble"
        info["algos_info"] = [algo._get_info() for algo in self._algos]
        return info
    