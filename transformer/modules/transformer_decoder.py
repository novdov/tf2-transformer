import tensorflow as tf

from transformer.modules import model_ops as ops
from transformer.modules.layers import DecoderLayer


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, num_heads, d_model, d_ff, decoder_vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = ops.create_embedding(decoder_vocab_size, d_model)
        self.position_encoding = ops.position_encoding(decoder_vocab_size, d_model)

        self.decoder_layers = [
            DecoderLayer(num_heads, d_model, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, decoder_input, encoder_output, mask, subsequent_mask, training):
        seq_len = tf.shape(decoder_input)[1]
        attention_weights = {}

        decoder_input = self.embedding(decoder_input)
        decoder_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        decoder_input += self.position_encoding[:, :seq_len, :]
        decoder_input = self.dropout(decoder_input, training=training)

        for i in range(self.num_layers):
            decoder_input, att_weights1, att_weights2 = self.decoder_layers[i](
                decoder_input, encoder_output, mask, subsequent_mask, training
            )
            attention_weights[f"decoder_layer{i+1}_weights1"] = att_weights1
            attention_weights[f"decoder_layer{i+1}_weights2"] = att_weights2

        decoder_output = decoder_input
        return decoder_output, attention_weights
