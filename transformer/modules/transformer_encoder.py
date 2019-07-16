import tensorflow as tf

from transformer.modules.layers import EncoderLayer
from transformer.utils import model_ops as ops


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, num_heads, d_model, d_ff, encoder_vocab_size, dropout_rate
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = ops.create_embedding(encoder_vocab_size, d_model)
        self.position_encoding = ops.position_encoding(encoder_vocab_size, d_model)

        self.encoder_layers = [
            EncoderLayer(num_heads, d_model, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_input, mask, training):
        seq_len = tf.shape(encoder_input)[1]

        encoder_input = self.embedding(encoder_input)
        encoder_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        encoder_input += self.position_encoding[:, :seq_len, :]
        encoder_input = self.dropout(encoder_input, training=training)

        for i in range(self.num_layers):
            encoder_input = self.encoder_layers[i](
                encoder_input, mask, training=training
            )
        encoder_output = encoder_input
        return encoder_output
