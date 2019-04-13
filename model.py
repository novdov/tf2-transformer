import tensorflow as tf

import model_ops as ops
from utils import logger


class Transformer(tf.keras.Model):

    def __init__(self, data, hparams, mode: str):
        super(Transformer, self).__init__()
        self.data = data
        self.hparams = hparams
        self.mode = mode

        self.embedding = ops.create_embedding(self.hparams.vocab_size,
                                              self.hparams.d_model)

    def encode(self, source_meta):
        hparams = self.hparams
        src_inputs, src_lens, sents1 = source_meta
        encoded = tf.nn.embedding_lookup(self.embedding, src_inputs)
        position = ops.position_encoding(encoded.get_shape()[0],
                                         hparams.d_moel)
        encoded += position
        if self.mode == "train":
            encoded = tf.nn.dropout(encoded, rate=self.hparams.dropout_rate)

        internal_dim = hparams.d_model // hparams.num_heads
        for idx in range(hparams.num_blocks):
            encoded = ops.multi_head_attention(
                query=encoded,
                keys=encoded,
                values=encoded,
                num_heads=hparams.num_heads,
                internal_dim=internal_dim,
                output_dim=hparams.d_model,
                name_prefix=f"blocks_{idx}"
            )
            encoded = ops.position_wise_feed_forward(
                encoded, num_units=[hparams.feed_forward_dim, hparams.d_model])
        memory = encoded
        return {
            "memory": memory,
            "sents1": sents1,
        }

    def decode(self, target_meta, memory):
        hparams = self.hparams
        decoder_input, tgt_sent, tgt_len, sents2 = target_meta
        decoded = tf.nn.embedding_lookup(self.embedding, decoder_input)
        position = ops.position_encoding(decoded.get_shape()[0],
                                         hparams.d_moel)
        decoded += position
        if self.mode == "train":
            decoded = tf.nn.dropout(decoded, rate=self.hparams.dropout_rate)

        dropout_rate = _get_dropout_rate(self.mode, hparams.dropout_rate)
        internal_dim = hparams.d_model // hparams.num_heads
        for idx in range(hparams.num_blocks):
            decoded = ops.multi_head_attention(
                query=decoded,
                keys=decoded,
                values=decoded,
                num_heads=hparams.num_heads,
                internal_dim=internal_dim,
                output_dim=hparams.d_model,
                name_prefix=f"blocks_{idx}",
                dropout_rate=dropout_rate,
                mask_subsequent=True,
            )
            decoded = ops.multi_head_attention(
                query=decoded,
                keys=memory,
                values=memory,
                num_heads=hparams.num_heads,
                internal_dim=internal_dim,
                output_dim=hparams.d_model,
                name_prefix=f"blocks_{idx}",
                dropout_rate=dropout_rate,
            )
            decoded = ops.position_wise_feed_forward(
                decoded, num_units=[hparams.feed_forward_dim, hparams.d_model])

        weights = tf.transpose(self.embedding)
        logits = tf.matmul(decoded, weights)
        prediction = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        return {
            "logits": logits,
            "pred": prediction,
            "target": tgt_sent,
            "sents2": sents2,
        }

    def train(self, source_meta, target_meta, step):
        hparams = self.hparams

        encoder_meta = self.encode(source_meta)
        memory, sents1 = encoder_meta["memory"], encoder_meta["sents1"]
        decoder_meta = self.decode(target_meta, memory)

        logits = decoder_meta["logits"]
        pred = decoder_meta["pred"]
        tgt_sent = decoder_meta["target"]
        sents2 = decoder_meta["sents2"]

        epsilon = hparams.epsilon
        logger.warn(f"Label smoothing with value {epsilon}. "
                    f"Don't use label smoothing in eval or prediction")
        labels = ops.label_smoothing(
            tf.one_hot(tgt_sent, depth=hparams.vocab_size))
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tgt_sent)

        learning_rate = ops.noam_lr_decay(hparams.learning_rate,
                                          step,
                                          hparams.learning_rate_warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        train_op = optimizer.minimize()



    def eval(self):
        pass


def _get_dropout_rate(mode, rate):
    if mode == "train":
        return rate
    return None
