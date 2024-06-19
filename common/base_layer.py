import tensorflow as tf


class BaseLayer(object):
    """Layer"""

    def __init__(self, name, activation=None, dropout=None, decay_mult=None):
        self._name = name
        self._activation = activation
        self._dropout = dropout
        self._decay_mult = decay_mult

    def get_variable(self, name, **kwargs):
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult
        return tf.compat.v1.get_variable(name, **kwargs)

    def __call__(self, *inputs):
        outputs = []
        for x in inputs:
            if type(x) == tuple or type(x) == list:
                y = self._forward(*x)
            else:
                y = self._forward(x)
            if self._activation:
                y = self._activation(y)
            if self._dropout:
                if hasattr(tf.flags.FLAGS, 'training'):
                    y = tf.cond(tf.flags.FLAGS.training,
                                lambda: tf.nn.dropout(y, keep_prob=1.0 - self._dropout),
                                lambda: y)
                else:
                    y = tf.nn.dropout(y, keep_prob=1.0 - self._dropout)
            outputs.append(y)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def _forward(self, x):
        return x 