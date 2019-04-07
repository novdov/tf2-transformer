import math
import tensorflow as tf


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
                         output_dim,
                         scope=None):
    reuse = tf.AUTO_REUSE
    attention_values = []

    with tf.variable_scope(scope), tf.name_scope(scope):
        for head_idx in range(1, num_heads+1):
            mapped_query = tf.layers.dense(
                inputs=query,
                units=internal_dim,
                activation=None,
                use_bias=False,
                name=f"head_{head_idx}/query",
                reuse=reuse
            )
            mapped_keys = tf.layers.dense(
                inputs=keys,
                units=internal_dim,
                activation=None,
                use_bias=False,
                name=f"head_{head_idx}/keys",
                reuse=reuse
            )
            mapped_values = tf.layers.dense(
                inputs=values,
                units=internal_dim,
                activation=None,
                use_bias=False,
                name=f"head_{head_idx}/values",
                reuse=reuse
            )

            att_values, alignments = scaled_dot_product_attention(
                mapped_query, mapped_keys, mapped_values)
            attention_values.append(att_values)

        outputs = tf.layers.dense(
            inputs=tf.concat(attention_values, axis=-1),
            units=output_dim,
            activation=None,
            name="concat_linear",
            reuse=reuse
        )
    return outputs


def layer_norm(inputs,
               do_scale=True,
               scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE), tf.name_scope(scope):
        length = inputs.get_shape()[-1]
        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        var = tf.reduce_mean(tf.square(inputs - mean),
                             axis=[-1],
                             keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(var + 1e12)

        if do_scale:
            scale = tf.get_variable(
                name="scale",
                shape=[length],
                initializer=tf.ones_initializer()
            )
            bias = tf.get_variable(
                name="bias",
                shape=[length],
                initializer=tf.zeros_initializer()
            )
            outputs = norm_inputs * scale + bias
        else:
            outputs = norm_inputs
    return outputs


def position_wise_feed_forward(inputs, num_units, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = layer_norm(outputs)
    return outputs


def position_embedding(length, depth):
    position = tf.to_float(tf.range(length))
    num_timescales = depth // 2

    log_timescale = math.log(10000.0) / (tf.to_float(num_timescales) - 1)
    div_terms = tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(div_terms, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def create_embedding(vocab_size, embedding_size, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(
            "word_embedding,",
            shape=[vocab_size, embedding_size],
            trainable=True,
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )
    return embedding
