import tensorflow as tf
from pathlib import Path


class Data:

    def __init__(self, hparams, mode: tf.estimator.ModeKeys):
        self.hparams = hparams
        self.mode = mode

    def create_iterator(self):
        def _create_iterator(filename):
            dataset = tf.data.TextLineDataset(filename)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                dataset.shuffle(10000, seed=1234)
            dataset.prefetch(500)
            dataset.batch(self.hparams.batch_size)
            return dataset.make_one_shot_iterator()

        data_dir = Path(__file__).resolve().parent / "iwslt2016/segmented/"
        filename_fmt = "{}.{}.bpe"

        return {
            "en": _create_iterator(
                data_dir.joinpath(filename_fmt.format(self.mode, "en"))),
            "de": _create_iterator(
                data_dir.joinpath(filename_fmt.format(self.mode, "de")))
        }
