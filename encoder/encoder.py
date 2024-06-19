from __future__ import absolute_import, division, print_function
import sys
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

from tensorflow.keras import Model
from common.layer_factory import LayerFactory
from tensorflow import keras
import tensorflow_addons as tf_ad
from common.crf import CRF

### 通用基类
class Encoder(Model):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config

    def call(self, inputs, training=None, mask=None):
        pass

    def build_graph(self, input_shape):
        '''自定义函数，在调用model.summary()之前调用
        '''
        self.build(input_shape)

        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name="inputs")
        _ = self.call(inputs)

### Classification task
class ClassifierEncoder(Encoder):
    def __init__(self, config):
        super(ClassifierEncoder, self).__init__(config)

        """@encoder_info.layers params examples:
           DNN: ["inputs", "embedding", "pools1d", "dense", "dense", "dense"]
           CNN: ["inputs", "embedding", "cnn"]
           LSTM: ["inputs", "embedding", "lstm", "pools1d"]
           Transformer: ["inputs", "embedding", "transformer", "dense", "pools1d"]
           Bert(Al-Bert): ["inputs", "albert", "lamda", "dense"]
        """
        self.encoder = LayerFactory(config).get_sequential_layer()
        self.classifier = Dense(config.intent_class_num, activation="softmax")

        self.encoder.build(input_shape=(None, config.max_seq_len))
        self.encoder.summary()

    def call(self, inputs, training=None, mask=None):
        encoded_output = self.encoder(inputs)
        logits = self.classifier(encoded_output)

        return logits


class NerEncoderV2(Encoder):
    def __init__(self, config):
        super(NerEncoderV2, self).__init__(config)

        self.encoder = LayerFactory(config).get_sequential_layer()
        self.dense_layer = Dense(config.tag_class_num, activation='softmax')

        self.crf = CRF(config.tag_class_num, name="crf_layer")
        self.encoder.add(self.dense_layer)
        self.encoder.add(self.crf)

        self.encoder.build(input_shape=(None, config.max_seq_len))
        self.encoder.summary()

    def call(self, inputs, training=None, mask=None):
        encoded_output = self.encoder(inputs)
        #logits = self.crf(self.dense_layer(encoded_output))
        logits = encoded_output
        return logits

### Ner task
class NerEncoder(Encoder):
    def __init__(self, config):
        super(NerEncoder, self).__init__(config)
        """@encoder_info.layers params examples:
                LSTM_crf: ["inputs", "embedding", "lstm", "dense"]
                Transformer_crf: ["inputs", "embedding", "transformer", "dense"]
                IDCNN_crf: ["inputs", "embedding", "idcnn"]
                Bert_crf: ["inputs", "albert", "lstm", "dense"]             
        """
        self.encoder = LayerFactory(config).get_sequential_layer()
        self.ner_dense_layer = Dense(config.tag_class_num, activation='softmax')

        self.transition_params = tf.Variable(tf.random.uniform(shape=(config.tag_class_num, config.tag_class_num)))

        self.encoder.build(input_shape=(None, config.max_seq_len))
        self.encoder.summary()

    def _forward(self, logits, text_lens, labels=None):
        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int64)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens, transition_params=self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens

    def call(self, inputs, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        logits = self.ner_dense_layer(self.encoder(inputs))

        return self._forward(logits, text_lens, labels)

### Joint-Learning(classification + Ner) task
class JointLearningEncoder(NerEncoder):
    def __init__(self, config):
        super(JointLearningEncoder, self).__init__(config)

        """@encoder_info.layers params examples:
                    LSTM: ["inputs", "embedding", "lstm", "dense"]
                    Transformer: ["inputs", "embedding", "transformer", "dense"]
                    Bert: ["inputs", "albert", "lstm"]             
        """
        self.pools1d = GlobalAveragePooling1D()
        self.classify_dense_layer = Dense(config.intent_class_num, activation="softmax")

    def call(self, inputs, tags=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)

        seq_out = self.encoder(inputs)
        classify_logits = self.classify_dense_layer(self.pools1d(seq_out))
        ner_logits = self.ner_dense_layer(seq_out)

        if tags is None:
            return classify_logits, ner_logits, text_lens

        ner_logits, text_lens, log_likelihood = super(JointLearningEncoder, self)._forward(ner_logits, text_lens, tags)
        return classify_logits, ner_logits, text_lens, log_likelihood


class TextMatchEncoder(Encoder):
    def __init__(self, config):
        super(TextMatchEncoder, self).__init__(config)

        self.encoder = LayerFactory(config).get_sequential_layer()

        self.encoder.build(input_shape=(None, config.max_seq_len))
        self.encoder.summary()

    def call(self, inputs, training=None, mask=None):
        encoded_output = self.encoder(inputs)

        return encoded_output
