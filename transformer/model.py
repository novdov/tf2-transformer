import tensorflow as tf

from transformer.modules import model_ops as ops
from transformer.modules.blocks import TransformerDecoder, TransformerEncoder
from transformer.utils.logging import logger


class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        encoder_vocab_size,
        decoder_vocab_size,
        dropout_rate=0.1,
    ):
        super().__init__(name="transformer")
        self.encoder = TransformerEncoder(
            num_layers, num_heads, d_model, d_ff, encoder_vocab_size, dropout_rate
        )
        logger.info("Building Encoder")
        self.decoder = TransformerDecoder(
            num_layers, num_heads, d_model, d_ff, decoder_vocab_size, dropout_rate
        )
        logger.info("Building Decoder")
        self.final_layer = tf.keras.layers.Dense(decoder_vocab_size)

    def call(self, features, training):
        source = features["inputs"]
        targets = features["targets"]

        enc_padding_mask, dec_padding_mask, combined_mask = ops.create_masks(
            source, targets
        )

        encoder_output = self.encoder(source, enc_padding_mask, training)
        decoder_output, attention_weights = self.decoder(
            targets, encoder_output, dec_padding_mask, combined_mask, training
        )
        final_output = self.final_layer(decoder_output)
        return final_output
