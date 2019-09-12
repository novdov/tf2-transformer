import math

import tensorflow as tf


def get_positional_encoding(length, d_model):
    position = tf.cast(tf.range(length), dtype=tf.float32)
    num_timescales = d_model // 2

    log_timescale = math.log(10000.0) / (tf.cast(num_timescales, dtype=tf.float32) - 1)
    div_terms = tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale
    )
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(div_terms, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal[tf.newaxis, :]


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = get_positional_encoding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        tf.float32

        x = self.token_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        return x
