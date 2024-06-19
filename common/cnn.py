from __future__ import absolute_import, division, print_function
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Concatenate, GlobalMaxPooling1D


class CNN(Model):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config

        self.conv1d = [Conv1D(config.num_filters, kernel_size, activation="relu") for kernel_size in config.kernel_sizes]
        self.pools1d = [GlobalMaxPooling1D() for _ in config.kernel_sizes]

    def call(self, emb, training=None, mask=None):
        conv1s = [self.pools1d[i](self.conv1d[i](emb)) for i in range(len(self.config.kernel_sizes))]
        out = Concatenate()(conv1s)

        return out