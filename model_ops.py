import math
import tensorflow as tf


def dense_layer(units, activation=None, use_bias=True, **kwargs):
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        **kwargs
    )


def scaled_dot_product_attention(query,
                                 keys,
                                 values,
                                 mask=None):
    """
    Calculate the attention weights.
    :param query: a query Tensor with shape of [..., seq_len_q, depth]
    :param keys: a key Tensor with shape of [..., seq_len_k, depth]
    :param values: a vale Tensor with shape of [..., seq_len_v, depth]
    :param mask: mask Tensor broadcastable
    :return: tuple of attention output and attention weights
    """
    d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_logits = tf.matmul(query, keys, transpose_b=True) * tf.math.rsqrt(d_k)

    if mask is not None:
        scaled_logits += (mask * 1e-9)

    weights = tf.nn.softmax(scaled_logits, axis=-1)
    output = tf.matmul(weights, values)
    return output, weights


def multi_head_attention(query,
                         keys,
                         values,
                         num_heads,
                         internal_dim,
                         output_dim,
                         name_prefix,
                         dropout_rate=None,
                         mask_subsequent=False):
    attention_values = []

    for head_idx in range(1, num_heads+1):
        mapped_query = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"{name_prefix}/head_{head_idx}/query"
        )(query)

        mapped_keys = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"{name_prefix}/head_{head_idx}/keys",
        )(keys)

        mapped_values = dense_layer(
            units=internal_dim,
            use_bias=False,
            name=f"{name_prefix}/head_{head_idx}/values",
        )(values)

        att_values, alignments = scaled_dot_product_attention(
            mapped_query, mapped_keys, mapped_values, dropout_rate, mask_subsequent)
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


def position_wise_feed_forward(d_ff, d_model):
    """
    Apply position wise feed forward to inputs.
    """
    return tf.keras.Sequential([
        dense_layer(units=d_ff, activation=tf.nn.relu),
        dense_layer(d_model)
    ])


def sublayer_connection(inputs, outputs):
    return layer_norm(inputs + outputs)


def position_encoding(length, depth):
    position = tf.cast(tf.range(length), dtype=tf.float32)
    num_timescales = depth // 2

    log_timescale = math.log(10000.0) / (tf.cast(num_timescales, dtype=tf.float32) - 1)
    div_terms = tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(div_terms, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return tf.expand_dims(signal, axis=0)


def create_embedding(vocab_size, embedding_size):
    return tf.keras.layers.Embedding(vocab_size, embedding_size)


def mask_tensor(input_tensor, subsequent=False):
    negative_inf = -2 ** 32 + 1
    if subsequent:
        diag_vals = tf.ones_like(tf.expand_dims(input_tensor[0], 0))
        triu = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        mask = tf.tile(triu, [input_tensor.shape[0], 1, 1])
    else:
        mask = tf.sign(input_tensor)

    paddings = tf.ones_like(mask) * negative_inf
    return tf.where(tf.equal(mask, 0), paddings, input_tensor)


def label_smoothing(input_tensor, epsilon=0.1):
    return (1 - epsilon) * input_tensor + (epsilon / input_tensor.shape[-1])


def noam_lr_decay(learning_rate, step, learning_rate_warmup_steps):
    learning_rate *= tf.minimum(
        step * learning_rate_warmup_steps ** -1.5,
        step ** -0.5
    )
    return learning_rate
