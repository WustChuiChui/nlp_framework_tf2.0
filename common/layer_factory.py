from __future__ import absolute_import, division, print_function
import sys
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional
from common.attention import Transformer
from common.cnn import CNN
from common.idcnn import IDCNN
from common.bert_factory import BertFactory
from embedding.word_embedding import *
from common.const import *
from tensorflow import keras
from common.crf import CRF

class LayerFactory():
    def __init__(self, config):
        self.config = config

    def add_single_layer(self, layer):
        if layer == "inputs":
            return tf.keras.layers.InputLayer(input_shape=(self.config.max_seq_len,), dtype=tf.int32, name="input_ids")
        elif layer == "embedding":
            return getattr(sys.modules[EMBEDDING_MODULE], self.config.embedding)(self.config)
        elif layer == "dense":
            return Dense(units=self.config.hidden_size, activation="relu")
        elif layer == "lamda":
            return keras.layers.Lambda(lambda x: x[:, 0, :])
        elif layer == "pools1d":
            return GlobalAveragePooling1D()
        elif layer == "cnn":
            return CNN(config=self.config)
        elif layer == "lstm":
            return Bidirectional(tf.keras.layers.LSTM(self.config.hidden_size, return_sequences=True))
        elif layer == "crf":
            return CRF(units=self.config.tag_class_num, name="crf_layer")
        elif layer == "idcnn":
            return IDCNN(self.config)
        elif layer == "transformer":
            return Transformer(self.config)
        elif layer == "albert":
            return BertFactory(config=self.config).load_albert_model()
        else:
            pass

    def get_sequential_layer(self):
        sequential = Sequential()
        for layer in self.config.layers:
            print("add layer: " + layer)
            sequential.add(self.add_single_layer(layer=layer))

        return sequential
