import tensorflow as tf


class ModelWeightsTool:
    
    def __init__(self, model):
        self._model = model
        self._phs = []  # placeholders
        self._assign_ops = []
        self._model_vars = model.variables()
        for v in self._model_vars:
            self._phs.append(tf.placeholder(tf.float32, shape=v.shape))
            self._assign_ops.append(tf.assign(v, self._phs[-1]))
    
    def set_weights(self, sess, weights):
        feed_dict = {}
        for i in range(len(self._phs)):
            feed_dict[self._phs[i]] = weights[i]
        sess.run(self._assign_ops, feed_dict)

    def get_weights(self, sess):
        return sess.run(self._model_vars)
