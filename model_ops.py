import math
import tensorflow as tf


def dense_layer(units, activation=None, use_bias=True, **kwargs):
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        **kwargs
    )


def scaled_dot_product_attention(query, keys, values):
    d_k = tf.shape(query)[-1]
    energy = tf.matmul(query, keys, transpose_b=True)
    alignments = tf.nn.softmax(energy * tf.rsqrt(tf.cast(d_k, tf.float32)))
    att_values = tf.matmul(alignments, values)
    return att_values, alignments


def multi_head_attention(query,
                         keys,
                         values,
                         num_heads,
                         internal_dim,
                         output_dim):
    attention_values = []

    for head_idx in range(1, num_heads+1):
        mapped_query = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"head_{head_idx}/query"
        )(query)

        mapped_keys = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"head_{head_idx}/keys",
        )(keys)

        mapped_values = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"head_{head_idx}/values",
        )(values)

        att_values, alignments = scaled_dot_product_attention(
            mapped_query, mapped_keys, mapped_values)
        attention_values.append(att_values)

    outputs = dense_layer(
        units=output_dim,
        name="concat_linear"
    )(tf.concat(attention_values, axis=-1))

    return outputs


def layer_norm(inputs, do_scale=True):
    length = inputs.get_shape()[-1]
    mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
    var = tf.reduce_mean(tf.square(inputs - mean),
                         axis=[-1],
                         keep_dims=True)
    norm_inputs = (inputs - mean) * tf.math.rsqrt(var + 1e12)

    if do_scale:
        scale = tf.Variable(initial_value=tf.ones(shape=[length]),
                            name="scale")
        bias = tf.Variable(initial_value=tf.zeros(shape=[length]),
                           name="bias")
        outputs = norm_inputs * scale + bias
    else:
        outputs = norm_inputs
    return outputs


def position_wise_feed_forward(inputs, num_units):
    outputs = dense_layer(units=num_units[0], activation=tf.nn.relu)(inputs)
    outputs = dense_layer(units=num_units[1])(outputs)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


def position_embedding(length, depth):
    """Position embedding from tensorflow official model."""
    position = tf.cast(tf.range(length), dtype=tf.float32)
    num_timescales = depth // 2

    log_timescale = math.log(10000.0) / (tf.cast(num_timescales, dtype=tf.float32) - 1)
    div_terms = tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(div_terms, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def create_embedding(vocab_size, embedding_size):
    embedding = tf.Variable(
        tf.initializers.GlorotUniform(shape=[vocab_size, embedding_size]),
        name="word_embedding",
        dtype=tf.float32
    )
    return embedding
