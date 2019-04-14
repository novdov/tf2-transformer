import tensorflow as tf

import model_ops as ops
from layers import Encoder, Decoder
from utils import logger


class Transformer(tf.keras.models.Model):

    def __init__(self,
                 num_layers,
                 num_heads,
                 d_model,
                 d_ff,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               num_heads,
                               d_model,
                               d_ff,
                               encoder_vocab_size,
                               dropout_rate)
        logger.info("Building Encoder")
        self.decoder = Decoder(num_layers,
                               num_heads,
                               d_model,
                               d_ff,
                               decoder_vocab_size,
                               dropout_rate)
        logger.info("Building Decoder")
        self.final_layer = ops.dense_layer(decoder_vocab_size)

    def call(self,
             inputs,
             targets,
             training,
             encoder_mask,
             decoder_mask,
             subsequent_mask):
        encoder_output = self.encoder(inputs, encoder_mask, training)
        decoder_output, attention_weights = self.decoder(
            targets, encoder_output, decoder_mask, subsequent_mask, training)
        final_output = self.final_layer(decoder_output)
        return final_output, attention_weights
