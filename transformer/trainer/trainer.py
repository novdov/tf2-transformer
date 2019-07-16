import os

import tensorflow as tf

from transformer.data.data import Data
from transformer.utils import model_ops as ops


def optimize_fn(model, loss, lr_schedule=None, metrics=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    metrics = metrics or [tf.keras.metrics.sparse_categorical_accuracy]
    return model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


class Trainer:
    def __init__(self, hparams, model):
        self.hparams = hparams
        self.model = model

        self.train_dataset = Data(hparams, mode="train")
        self.eval_dataset = Data(hparams, mode="eval")

        self.train_batches = self.train_dataset.batchify_data()
        self.eval_batches = self.eval_dataset.batchify_data()

    def train_and_evaluate(
        self, train_steps, eval_steps, eval_frequency, checkpoint_dir
    ):
        callbacks = [
            tf.keras.callbacks.History(),
            tf.keras.callbacks.BaseLogger(),
            tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir),
        ]

        last_epoch = 0
        output_fmt = os.path.join(checkpoint_dir, "model-{epoch:05d}")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_fmt, save_weights_only=True, period=5
            )
        )

        checkpoints = tf.io.gfile.glob(os.path.join(checkpoint_dir, "model-*"))
        checkpoints = [os.path.basename(ckpt)[6:] for ckpt in checkpoints]

        epoch_numbers = [int(ckpt[:5]) for ckpt in checkpoints if len(ckpt) > 4]
        epoch_numbers.sort()

        learning_rate = ops.CustomLearningRateSchedule(self.hparams.d_model)
        optimize_fn(self.model, loss=ops.loss_function, lr_schedule=learning_rate)

        if epoch_numbers:
            last_epoch = epoch_numbers[-1]
            saved_path = os.path.join(checkpoint_dir, f"model-{last_epoch:.05d}")
            self.model.load_weights(saved_path)

        self.model.fit(
            x=self.train_batches,
            epochs=train_steps // eval_frequency,
            # steps_per_epoch=eval_frequency,
            validation_data=self.eval_batches,
            validation_steps=eval_steps,
            initial_epoch=last_epoch,
            callbacks=callbacks,
        )
