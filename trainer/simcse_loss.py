import sys
import json

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.losses import *


class SimLoss(keras.losses.Loss):
    def __init__(self):
        super(SimLoss, self).__init__()

    def call(self, y_true, y_pred):
        idxs = tf.range(0, tf.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.keras.backend.floatx())
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
        similarities = similarities / 0.05
        loss = categorical_crossentropy(y_true, similarities, from_logits=True)

        return tf.reduce_mean(loss)


class SimHardLoss(keras.losses.Loss):
    def __init__(self):
        super(SimHardLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.range(0, tf.shape(y_pred)[0])
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        similarities = tf.matmul(y_pred, y_pred, transpose_b=True)  # batch_size * batch_size

        # 筛选负样本(选出距离最近的负样本)
        sim_idx = tf.argsort(similarities, axis=-1)
        row = tf.gather(sim_idx, tf.shape(y_pred)[0] - 2, axis=1)  # 此处取一个与当前样本距离最近的负样本
        # 筛选正样本
        col = tf.range(tf.shape(y_pred)[0])

        similarities = tf.gather(similarities, row, axis=0)  # 筛选负样本
        similarities = tf.gather(similarities, col, axis=1)  # 筛选正样本
        similarities = similarities / 0.05
        loss = sparse_categorical_crossentropy(y_true, similarities, from_logits=True)

        return tf.reduce_mean(loss)
