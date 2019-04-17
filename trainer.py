import os
import tensorflow as tf

import model_ops as ops
from data import Data


def optimize_fn(model,
                loss,
                lr_schedule=None,
                metrics=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    metrics = metrics or [tf.keras.metrics.sparse_categorical_accuracy]
    return model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)


def train_fn(hparams,
             output_dir,
             model,
             train_steps,
             eval_steps,
             eval_frequency):

    learning_rate = ops.CustomAdamLearningRateSchedule(hparams.d_model)
    optimize_fn(model,
                loss=ops.loss_function,
                lr_schedule=learning_rate)

    train_dataset = Data(hparams, mode="train")
    eval_dataset = Data(hparams, mode="eval")

    train_batches = train_dataset.batchify_data()
    eval_batches = eval_dataset.batchify_data()

    # model.fit(train_batches, epochs=1, steps_per_epoch=1)

    callbacks = [
        tf.keras.callbacks.History(),
        tf.keras.callbacks.BaseLogger(),
        tf.keras.callbacks.TensorBoard(log_dir=output_dir)
    ]
    last_epoch = 0
    output_fmt = os.path.join(output_dir, "model-{epoch:05d}")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=output_fmt, save_weights_only=True, period=5))
    checkpoints = tf.io.gfile.glob(os.path.join(output_dir, "model-*"))
    checkpoints = [os.path.basename(ckpt)[6:] for ckpt in checkpoints]
    epoch_numbers = [int(ckpt[:5]) for ckpt in checkpoints if len(ckpt) > 4]
    epoch_numbers.sort()
    if epoch_numbers:
        last_epoch = epoch_numbers[-1]
        saved_path = os.path.join(output_dir, f"model-{last_epoch:.05d}")
        model.load_weights(saved_path)

    model.fit(x=train_batches,
              epochs=train_steps // eval_frequency,
              # steps_per_epoch=eval_frequency,
              validation_data=eval_batches,
              validation_steps=eval_steps,
              initial_epoch=last_epoch,
              callbacks=callbacks)
