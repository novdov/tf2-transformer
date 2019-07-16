import tensorflow as tf

from transformer.modules.multi_head_attention import MultiHeadAttention
from transformer.utils import model_ops as ops


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = ops.position_wise_feed_forward(d_ff, d_model)

        self.att_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ff_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_inputs, mask, training):
        att_output, _ = self.multi_head_attention(
            encoder_inputs, encoder_inputs, encoder_inputs, mask
        )
        att_output = self.att_dropout(att_output, training=training)
        output1 = ops.sublayer_connection(encoder_inputs, att_output)

        ff_output = self.ffn(output1)
        ff_output = self.ff_dropout(ff_output, training=training)
        output2 = ops.sublayer_connection(output1, ff_output)
        return output2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = ops.position_wise_feed_forward(d_ff, d_model)

        self.att_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.att_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.ff_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, decoder_inputs, encoder_output, mask, subsequent_mask, training):
        att1, att_weights1 = self.multi_head_attention1(
            decoder_inputs, decoder_inputs, decoder_inputs, subsequent_mask
        )
        att1 = self.att_dropout1(att1, training=training)
        output1 = ops.sublayer_connection(decoder_inputs, att1)

        att2, att_weights2 = self.multi_head_attention2(
            output1, encoder_output, encoder_output, mask
        )
        att2 = self.att_dropout2(att2, training=training)
        output2 = ops.sublayer_connection(output1, att2)

        ff_output = self.ffn(output2)
        ff_output = self.ff_dropout(ff_output, training=training)
        output3 = ops.sublayer_connection(output2, ff_output)

        return output3, att_weights1, att_weights2
