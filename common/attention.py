from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

class Transformer(layers.Layer):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        assert config.dim_model % config.num_heads == 0

        self.depth = config.dim_model // config.num_heads

        self.wq = layers.Dense(config.dim_model)
        self.wk = layers.Dense(config.dim_model)
        self.wv = layers.Dense(config.dim_model)

        self.dense = layers.Dense(config.dim_model)

    def get_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        return seq[:, np.newaxis, np.newaxis, :] # [batch_size, 1, 1, seq_len]

    def attention(self, Q, K, V, mask=None):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_att_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_att_logits += (mask * -1e9)

        att_weights = tf.nn.softmax(scaled_att_logits, axis=-1)
        out = tf.matmul(att_weights, V)

        return out, att_weights

    def split_heads(self, x):
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.config.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]

        q = self.split_heads(self.wq(inputs))
        k = self.split_heads(self.wk(inputs))
        v = self.split_heads(self.wv(inputs))

        scaled_attention, att_weights = self.attention(q, k, v, mask)
        concat_att = tf.reshape(scaled_attention, (batch_size, -1, self.config.dim_model))

        out = self.dense(concat_att)
        return out