import tensorflow as tf

from .embedding import TransformerEmbedding
from .layers import TransformerDecoderLayer, TransformerEncoderLayer


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, num_heads, d_model, d_ff, encoder_vocab_size, dropout_rate
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = TransformerEmbedding(encoder_vocab_size, d_model)

        self.encoder_layers = [
            TransformerEncoderLayer(num_heads, d_model, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_input, mask, training):
        encoder_input = self.embedding(encoder_input, training)
        for i in range(self.num_layers):
            encoder_input = self.encoder_layers[i](
                encoder_input, mask, training=training
            )
        encoder_output = encoder_input
        return encoder_output


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, num_heads, d_model, d_ff, decoder_vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = TransformerEmbedding(decoder_vocab_size, d_model)
        self.decoder_layers = [
            TransformerDecoderLayer(num_heads, d_model, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, decoder_input, encoder_output, mask, subsequent_mask, training):
        attention_weights = {}

        decoder_input = self.embedding(decoder_input, training)
        for i in range(self.num_layers):
            decoder_input, att_weights1, att_weights2 = self.decoder_layers[i](
                decoder_input, encoder_output, mask, subsequent_mask, training
            )
            attention_weights[f"decoder_layer{i+1}_weights1"] = att_weights1
            attention_weights[f"decoder_layer{i+1}_weights2"] = att_weights2

        decoder_output = decoder_input
        return decoder_output, attention_weights
