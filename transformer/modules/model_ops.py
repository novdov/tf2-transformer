import tensorflow as tf


def dense_layer(units, activation=None, use_bias=True, **kwargs):
    return tf.keras.layers.Dense(
        units=units, activation=activation, use_bias=use_bias, **kwargs
    )


def create_padding_mask(input_tensor):
    sequence = tf.cast(tf.math.equal(input_tensor, 0), tf.float32)
    return sequence[:, tf.newaxis, tf.newaxis, :]


def create_subsequent_mask(length):
    return 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)


def create_masks(inputs, targets):
    enc_padding_mask = create_padding_mask(inputs)
    dec_padding_mask = create_padding_mask(inputs)
    subsequent_mask = create_subsequent_mask(tf.shape(targets)[1])
    dec_target_padding_mask = create_padding_mask(targets)
    combined_mask = tf.maximum(dec_target_padding_mask, subsequent_mask)
    return enc_padding_mask, dec_padding_mask, combined_mask


def label_smoothing(input_tensor, epsilon=0.1):
    return (1 - epsilon) * input_tensor + (epsilon / input_tensor.shape[-1])


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)


def loss_function(true, pred):
    object_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.not_equal(true, 0)
    loss = object_fn(true, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)
