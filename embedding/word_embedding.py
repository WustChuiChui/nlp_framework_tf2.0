import logging

import tensorflow as tf
from common.base_layer import BaseLayer
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Embedding
from tensorflow.python.ops import embedding_ops
import numpy as np
class WordEmbedding(Embedding):
    def __init__(self, config, **kwargs):
        self.embeddings = Embedding.__init__(self, input_dim=config.vocab_size, output_dim=config.embedding_dims, input_length=config.max_seq_len)

    def _forward(self, inputs):
        dtype = backend.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup_v2(self.embeddings, inputs)

        return out

class RegionAlignmentLayer(BaseLayer):
    def __init__(self, config, **args):
        BaseLayer.__init__(self, config.embedding, **args)
        self._region_size = config.region_size if hasattr(config, "region_size") else 3

    def _forward(self, x):
        region_radius = int(self._region_size / 2)
        aligned_seq = list(map(lambda i: tf.slice(x, [0, i - region_radius], [-1, self._region_size]), \
                range(region_radius, x.shape[1] - region_radius)))
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2])
        return aligned_seq

class WinPoolEmbedding(WordEmbedding):
    """WindowPoolEmbeddingLayer"""

    def __init__(self, config, **kwargs):
        BaseLayer.__init__(self, config.embedding, **kwargs)
        self.region_alignment_layer = RegionAlignmentLayer(config)
        super(WinPoolEmbedding, self).__init__(config, **kwargs)

    def _forward(self, seq):
        region_aligned_seq = self.region_alignment_layer(seq)
        region_aligned_emb = super(WinPoolEmbedding, self)._forward(region_aligned_seq)

        return tf.reduce_max(region_aligned_emb, axis=2)

class ScalarRegionEmbedding(WordEmbedding):
    def __init__(self, config, **kwargs):
        BaseLayer.__init__(self, config.embedding, **kwargs)
        self.region_alignment_layer = RegionAlignmentLayer(config)
        super(ScalarRegionEmbedding, self).__init__(config, **kwargs)

    def _forward(self, seq):
        region_aligned_seq = self.region_alignment_layer(seq)
        region_aligned_emb = super(ScalarRegionEmbedding, self)._forward(region_aligned_seq)

        region_radius = int(self.region_alignment_layer._region_size / 2)
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = embedding_ops.embedding_lookup_v2(self.embeddings, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return tf.reduce_max(projected_emb, axis=2)

class EnhancedWordEmbedding(WordEmbedding):
    """EnhancedWordEmbedding"""

    def __init__(self, config, **kwargs):
        super(EnhancedWordEmbedding, self).__init__(config, **kwargs)

        self.scale = config.scale if hasattr(config, "scale") else 0.5
        self.emb_size = config.embedding_dims

    def _forward(self, seq):
        outputs = super(EnhancedWordEmbedding, self)._forward(seq)

        outputs = outputs * (self.emb_size ** self.scale)
        return outputs

class ContextWordRegionEmbedding(WordEmbedding):
    """ContextWordRegionEmbedding"""
    def __init__(self, config, **kwargs):
        self._region_size = config.region_size if hasattr(config, "region_size") else 3
        super(ContextWordRegionEmbedding, self).__init__(config, **kwargs)

        self._unit_id_bias = np.array([i * config.vocab_size for i in range(self._region_size)])
        self.region_alignment_layer = RegionAlignmentLayer(config)

    def _region_aligned_units(self, seq):
        """
        _region_aligned_unit
        """
        region_aligned_seq = self.region_alignment_layer(seq) + self._unit_id_bias
        region_aligned_unit = super(ContextWordRegionEmbedding, self)._forward(region_aligned_seq)

        return region_aligned_unit

    def _forward(self, seq):
        """forward
        """
        region_radius = int(self._region_size / 2)
        word_emb = embedding_ops.embedding_lookup_v2(self.embeddings,
                                tf.slice(seq, [0, region_radius], \
                                [-1, tf.cast(seq.get_shape()[1] - 2 * region_radius, tf.int32)]))

        word_emb = tf.expand_dims(word_emb, 2)   #[batch_size, seq_len - 2 * region_radius, 1, embedd_dim]
        region_aligned_unit = self._region_aligned_units(seq)  #[batch_size, seq_len - 2 * region_radius, 3, embedd_dim]
        embedding = region_aligned_unit * word_emb   #[batch_size, seq_len - 2 * region_radius, 1, embedd_dim]
        embedding = tf.reduce_max(embedding, axis=2)   #[batch_size, seq_len - 2 * region_radius, embedd_dim]
        return embedding

class WordContextRegionEmbedding(WordEmbedding):
    """WordContextRegionEmbedding"""
    def __init__(self, config, **kwargs):
        BaseLayer.__init__(self, config.embedding, **kwargs)
        self._region_size = config.region_size if hasattr(config, "region_size") else 3
        self._region_alignment_layer = RegionAlignmentLayer(config)
        super(WordContextRegionEmbedding, self).__init__(config, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = self._region_alignment_layer(seq)
        region_aligned_emb = super(WordContextRegionEmbedding, self)._forward(region_aligned_seq)

        region_radius = int(self._region_size / 2)
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = embedding_ops.embedding_lookup_v2(self.embeddings, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return tf.reduce_max(projected_emb, axis=2)
