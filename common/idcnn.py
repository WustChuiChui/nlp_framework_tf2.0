from __future__ import absolute_import, division, print_function
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

class IDCNN(Model):
    def __init__(self, config):
        super(IDCNN, self).__init__()
        self.config = config

        self.conv2d = Conv2D(config.num_filters, (1, config.filter_width), padding="same")

        self.w = tf.Variable(
            tf.compat.v1.random.truncated_normal(shape=[1, config.filter_width, config.num_filters, config.num_filters], stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")

    def call(self, emb, training=None, mask=None):
        layer_input = self.conv2d(tf.expand_dims(emb, 1))

        final_layers = []
        for j in range(self.config.repeat_times):
            for i in range(len(self.config.num_layers)):
                conv = tf.nn.atrous_conv2d(layer_input, self.w, rate=self.config.num_layers[i], padding="SAME")
                conv = tf.nn.relu(tf.nn.bias_add(conv, self.b))
                if i == (len(self.config.num_layers) - 1):
                    final_layers.append(conv)
                layer_input = conv

        out = tf.squeeze(tf.concat(axis=3, values=final_layers), [1])
        return out