import tensorflow as tf


def scaled_dot_product_attention(query, keys, values, mask=None):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    :param query: a query Tensor with shape of [..., seq_len_q, depth]
    :param keys: a key Tensor with shape of [..., seq_len_k, depth]
    :param values: a vale Tensor with shape of [..., seq_len_v, depth_v]
    :param mask: mask Tensor broadcastable
    :return: tuple of attention output and attention weights
    """
    d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
    # (batch_size, seq_len_q, seq_len_k)
    scaled_logits = tf.matmul(query, keys, transpose_b=True) * tf.math.rsqrt(d_k)

    if mask is not None:
        scaled_logits += mask * -1e9

    # (batch_size, seq_len_q, seq_len_k)
    weights = tf.nn.softmax(scaled_logits, axis=-1)
    # (batch_size, seq_len_q, depth_v)
    output = tf.matmul(weights, values)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.depth = self.d_model // self.num_heads

        self.query_proj = tf.keras.layers.Dense(self.d_model)
        self.key_proj = tf.keras.layers.Dense(self.d_model)
        self.value_proj = tf.keras.layers.Dense(self.d_model)

        self.fc = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, input_tensor, batch_size):
        input_tensor = tf.reshape(
            input_tensor, [batch_size, -1, self.num_heads, self.depth]
        )
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, -1, depth)
        return tf.transpose(input_tensor, perm=[0, 2, 1, 3])

    def call(self, query, keys, values, mask):
        # query, key, values: (batch_size, seq_len, d_model)
        batch_size = tf.shape(query)[0]

        # projected q, k, v -> (batch_size, seq_len, d_model)
        # mapped q, k, v -> (batch_size, num_heads, seq_len_q, depth)
        mapped_query = self.split_heads(self.query_proj(query), batch_size)
        mapped_keys = self.split_heads(self.key_proj(keys), batch_size)
        mapped_values = self.split_heads(self.value_proj(values), batch_size)

        # att_output.shape -> (batch_size, num_heads, seq_len_q, depth)
        # att_weights.shape -> (batch_size, num_heads, seq_len_q, seq_len_k)
        att_output, att_weights = scaled_dot_product_attention(
            mapped_query, mapped_keys, mapped_values, mask
        )
        # (batch_size, seq_len_q, num_heads, depth)
        att_output = tf.transpose(att_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_att = tf.reshape(att_output, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_v, d_model)
        output = self.fc(concat_att)
        return output, att_weights
