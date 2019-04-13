import tensorflow as tf

import model_ops as ops


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.depth = self.d_model // self.num_heads

    def split_heads(self, input_tensor, batch_size):
        input_tensor = tf.reshape(input_tensor,
                                  [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(input_tensor, perm=[0, 2, 1, 3])

    def call(self, query, keys, values, mask):
        batch_size = tf.shape(query)[0]

        mapped_query = self.split_heads(ops.dense_layer(self.d_model)(query),
                                        batch_size)
        mapped_keys = self.split_heads(ops.dense_layer(self.d_model)(keys),
                                       batch_size)
        mapped_values = self.split_heads(ops.dense_layer(self.d_model)(values),
                                         batch_size)

        att_output, att_weights = ops.scaled_dot_product_attention(
            mapped_query, mapped_keys, mapped_values, mask)
        # (batch_size, seq_len_v, num_heads, depth)
        att_output = tf.transpose(att_output, perm=[0, 2, 1, 3])

        concat_att = tf.reshape(att_output, (batch_size, -1, self.d_model))
        output = ops.dense_layer(self.d_model)(concat_att)
        return output, att_weights
