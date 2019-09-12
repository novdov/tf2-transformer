import tensorflow as tf

from .attention import MultiHeadAttention


class PointWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

    def __call__(self, ffn_input):
        return self.ffn(ffn_input)


class SublayerConnection(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForward(d_ff, d_model)
        self.sublayers = [SublayerConnection() for _ in range(2)]

        self.att_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_inputs, mask, training):
        att_output, _ = self.multi_head_attention(
            encoder_inputs, encoder_inputs, encoder_inputs, mask
        )
        att_output = self.att_dropout(att_output, training=training)
        output1 = self.sublayers[0](encoder_inputs, att_output)

        ffn_output = self.ffn(output1)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        output2 = self.sublayers[1](output1, ffn_output)
        return output2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super().__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForward(d_ff, d_model)
        self.sublayers = [SublayerConnection() for _ in range(3)]

        self.att_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.att_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, decoder_inputs, encoder_output, mask, subsequent_mask, training):
        att1, att_weights1 = self.multi_head_attention1(
            decoder_inputs, decoder_inputs, decoder_inputs, subsequent_mask
        )
        att1 = self.att_dropout1(att1, training=training)
        output1 = self.sublayers[0](decoder_inputs, att1)

        att2, att_weights2 = self.multi_head_attention2(
            output1, encoder_output, encoder_output, mask
        )
        att2 = self.att_dropout2(att2, training=training)
        output2 = self.sublayers[1](output1, att2)

        ffn_output = self.ffn(output2)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        output3 = self.sublayers[2](output2, ffn_output)

        return output3, att_weights1, att_weights2
