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


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = ops.position_wise_feed_forward(d_ff, d_model)

        self.att_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ff_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_inputs, mask, training):
        att_output, _ = self.multi_head_attention(
            encoder_inputs, encoder_inputs, encoder_inputs, mask)
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
            decoder_inputs,
            decoder_inputs,
            decoder_inputs,
            subsequent_mask
        )
        att1 = self.att_dropout1(att1, training=training)
        output1 = ops.sublayer_connection(decoder_inputs, att1)

        att2, att_weights2 = self.multi_head_attention2(
            output1,
            encoder_output,
            encoder_output,
            mask
        )
        att2 = self.att_dropout2(att2, training=training)
        output2 = ops.sublayer_connection(output1, att2)

        ff_output = self.ffn(output2)
        ff_output = self.ff_dropout(ff_output, training=training)
        output3 = ops.sublayer_connection(output2, ff_output)

        return output3, att_weights1, att_weights2


class Encoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 num_heads,
                 d_model,
                 d_ff,
                 encoder_vocab_size,
                 dropout_rate):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = ops.create_embedding(encoder_vocab_size, d_model)
        self.position_encoding = ops.position_encoding(encoder_vocab_size, d_model)

        self.encoder_layers = [EncoderLayer(num_heads, d_model, d_ff, dropout_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, encoder_input, mask, training):
        seq_len = tf.shape(encoder_input)[1]

        encoder_input = self.embedding(encoder_input)
        encoder_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        encoder_input += self.position_encoding[:, :seq_len, :]
        encoder_input = self.dropout(encoder_input, training=training)

        for i in range(self.num_layers):
            encoder_input = self.encoder_layers[i](encoder_input,
                                                   mask,
                                                   training=training)
        encoder_output = encoder_input
        return encoder_output


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 num_heads,
                 d_model,
                 d_ff,
                 decoder_vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = ops.create_embedding(decoder_vocab_size, d_model)
        self.position_encoding = ops.position_encoding(decoder_vocab_size, d_model)

        self.decoder_layers = [DecoderLayer(num_heads, d_model, d_ff, dropout_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             decoder_input,
             encoder_output,
             mask,
             subsequent_mask,
             training):
        seq_len = tf.shape(decoder_input)[1]
        attention_weights = {}

        decoder_input = self.embedding(decoder_input)
        decoder_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        decoder_input += self.position_encoding[:, :seq_len, :]
        decoder_input = self.dropout(decoder_input, training=training)

        for i in range(self.num_layers):
            decoder_input, att_weights1, att_weights2 = self.decoder_layers[i](
                decoder_input, encoder_output, mask, subsequent_mask, training)
            attention_weights[f"decoder_layer{i+1}_weights1"] = att_weights1
            attention_weights[f"decoder_layer{i+1}_weights2"] = att_weights2

        decoder_output = decoder_input
        return decoder_output, attention_weights
