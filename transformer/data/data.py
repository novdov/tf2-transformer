import os
from pathlib import Path

import tensorflow as tf


class Data:
    def __init__(self, hparams, mode: str):
        self.hparams = hparams
        self.mode = mode

        self.data_dir = Path(__file__).parents[2].joinpath("iwslt2016/segmented/")
        self.filename_fmt = "{}.{}.bpe"
        self.token2id, self.id2token = self._load_vocab()

    def batchify_data(self):
        types = ({"inputs": tf.int32, "targets": tf.int32}, tf.int32)
        shapes = ({"inputs": [None], "targets": [None]}, [None])
        batch_size = self.hparams.batch_size

        dataset = tf.data.TextLineDataset.from_generator(
            self._create_generator, output_types=types, output_shapes=shapes
        )
        if self.mode == "train":
            dataset.shuffle(128)

        dataset = dataset.repeat(self.hparams.eval_frequency)
        dataset = dataset.padded_batch(batch_size, shapes).prefetch(8)
        return dataset

    def _create_generator(self):
        sents1, sents2 = self._load_data()
        for sent1, sent2 in zip(sents1, sents2):
            inputs_encoded = self._encode(sent1, "source")
            targets_encoded = self._encode(sent2, "target")

            inputs = {"inputs": inputs_encoded, "targets": targets_encoded[:-1]}
            targets_real = targets_encoded[1:]

            yield inputs, targets_real

    def _load_data(self):
        max_len = self.hparams.max_len
        source_path = os.path.join(
            self.data_dir, self.filename_fmt.format(self.mode, "en")
        )
        target_path = os.path.join(
            self.data_dir, self.filename_fmt.format(self.mode, "de")
        )

        sents1, sents2 = [], []
        with tf.io.gfile.GFile(source_path) as f_source, tf.io.gfile.GFile(
            target_path
        ) as f_target:
            for sent1, sent2 in zip(f_source, f_target):
                if len(sent1.split()) + 1 > max_len:
                    continue
                if len(sent2.split()) + 1 > max_len:
                    continue
                sents1.append(sent1)
                sents2.append(sent2)
        return sents1, sents2

    def _load_vocab(self):
        vocab = [
            line.split()[0]
            for line in open(self.hparams.vocab_path, "r").read().splitlines()
        ]
        token2id = {token: idx for idx, token in enumerate(vocab)}
        id2token = {token: idx for idx, token in enumerate(vocab)}
        return token2id, id2token

    def _encode(self, sentence, type_):
        # sentence = sentence.decode("utf-8")
        if type_ == "source":
            tokens = sentence.split() + ["</s>"]
        elif type_ == "target":
            tokens = ["<s>"] + sentence.split() + ["</s>"]
        else:
            raise ValueError

        unk = self.token2id["<unk>"]
        return [self.token2id.get(token, unk) for token in tokens]
