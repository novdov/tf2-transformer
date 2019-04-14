import os
import re
import sentencepiece as spm
from tqdm import tqdm
from pathlib import Path
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from utils import logger


class Preprocessor:

    def __init__(self, hparams):
        self.hparams = hparams
        self.file_paths = {
            "train": {"en": "iwslt2016/train.tags.de-en.en",
                      "de": "iwslt2016/train.tags.de-en.de"},
            "eval": {"en": "iwslt2016/IWSLT16.TED.tst2013.de-en.en.xml",
                     "de": "iwslt2016/IWSLT16.TED.tst2013.de-en.de.xml"},
            "test": {"en": "iwslt2016/IWSLT16.TED.tst2014.de-en.en.xml",
                     "de": "iwslt2016/IWSLT16.TED.tst2014.de-en.de.xml"}
        }
        self.files = {key: {} for key in self.file_paths.keys()}
        for mode, file_paths in self.file_paths.items():
            for lang, file_path in file_paths.items():
                self.files[mode][lang] = read_file(file_path, mode)

        self.data_output_fmt = "iwslt2016/{}/"
        self.write_preprocessed_file()

    def load_tokenizer(self, model_path):
        self.sentpiece_processor = spm.SentencePieceProcessor()
        self.sentpiece_processor.Load(model_path)

    def train_tokenizer(self, model_prefix):
        input_path = os.path.join(self.data_output_fmt.format("preprocessed"), "train")
        params = f"--input={input_path} --pad_id=0 " \
                 f"--unk_id=1 --bos_id=2 --eos_id=3 " \
                 f"--model_prefix={model_prefix} " \
                 f"--vocab_size={self.hparams.vocab_size} " \
                 f"--model_type=bpe"
        spm.SentencePieceTrainer.Train(params)

    def segment_and_write(self):
        def _segment_and_write(sents, filename):
            with open(filename, "w") as f:
                for sent in sents:
                    pieces = self.sentpiece_processor.EncodeAsPieces(sent)
                    f.write(" ".join(pieces) + "\n")

        output_dir = self.data_output_fmt.format("segmented")
        os.makedirs(output_dir, exist_ok=True)
        for mode, files in self.files.items():
            for lang, data in files.items():
                file_path = f"{mode}.{lang}.bpe"
                logger.info(f"Write segmented file at {file_path}.")
                _segment_and_write(data, os.path.join(output_dir, file_path))

    def write_preprocessed_file(self):
        def _write_file(sents, filename):
            with open(filename, "w") as f:
                f.write("\n".join(sents))

        output_dir = self.data_output_fmt.format("preprocessed")
        os.makedirs(output_dir, exist_ok=True)
        for mode, files in self.files.items():
            for lang, data in files.items():
                file_path = f"{mode}.{lang}"
                logger.info(f"Write preprocessed file at {file_path}.")
                _write_file(data, os.path.join(output_dir, file_path))

        all_train_data = self.files["train"]["en"] + self.files["train"]["de"]
        _write_file(all_train_data, os.path.join(output_dir, "train"))


def read_file(filename, mode):
    logger.info(f"Read file from {filename}.")
    if mode == "train":
        texts = [line for line in tqdm(Path(filename).open("r").read().splitlines())
                 if not line.startswith("<")]
    else:
        texts = [re.sub("<[^>]+>", "", line)
                 for line in tqdm(Path(filename).open("r").read().splitlines())
                 if line.startswith("<seq")]
    return texts
