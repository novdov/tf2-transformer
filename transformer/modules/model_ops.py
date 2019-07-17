import math

import tensorflow as tf


def dense_layer(units, activation=None, use_bias=True, **kwargs):
    return tf.keras.layers.Dense(
        units=units, activation=activation, use_bias=use_bias, **kwargs
    )


def scaled_dot_product_attention(query, keys, values, mask=None):
    """
    Calculate the attention weights.
    :param query: a query Tensor with shape of [..., seq_len_q, depth]
    :param keys: a key Tensor with shape of [..., seq_len_k, depth]
    :param values: a vale Tensor with shape of [..., seq_len_v, depth]
    :param mask: mask Tensor broadcastable
    :return: tuple of attention output and attention weights
    """
    d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
    # (batch_size, seq_len_q, seq_len_k)
    scaled_logits = tf.matmul(query, keys, transpose_b=True) * tf.math.rsqrt(d_k)

    if mask is not None:
        scaled_logits += mask * 1e-9

    # (batch_size, seq_len_q, seq_len_k)
    weights = tf.nn.softmax(scaled_logits, axis=-1)
    # (batch_size, seq_len_q, depth_v)
    output = tf.matmul(weights, values)
    return output, weights


def position_encoding(length, depth):
    position = tf.cast(tf.range(length), dtype=tf.float32)
    num_timescales = depth // 2

    log_timescale = math.log(10000.0) / (tf.cast(num_timescales, dtype=tf.float32) - 1)
    div_terms = tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale
    )
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(div_terms, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal[tf.newaxis, :]


def create_embedding(vocab_size, embedding_size):
    return tf.keras.layers.Embedding(vocab_size, embedding_size)


def create_padding_mask(input_tensor):
    sequence = tf.cast(tf.math.equal(input_tensor, 0), tf.float32)
    return sequence[:, tf.newaxis, tf.newaxis, :]


def create_subsequent_mask(length):
    return 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)


def create_masks(inputs, targets):
    enc_padding_mask = create_padding_mask(inputs)
    dec_padding_mask = create_padding_mask(inputs)
    subsequent_mask = create_subsequent_mask(tf.shape(targets)[1])
    dec_target_padding_mask = create_padding_mask(targets)
    combined_mask = tf.maximum(dec_target_padding_mask, subsequent_mask)
    return enc_padding_mask, dec_padding_mask, combined_mask


def label_smoothing(input_tensor, epsilon=0.1):
    return (1 - epsilon) * input_tensor + (epsilon / input_tensor.shape[-1])


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)


def loss_function(true, pred):
    object_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.not_equal(true, 0)
    loss = object_fn(true, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)
