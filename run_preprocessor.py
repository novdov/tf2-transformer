from pathlib import Path

from preprocessor import Preprocessor
from hparam_utils import Hparams


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    hparams = Hparams(base_dir.joinpath("hparams.json"))

    preprocessor = Preprocessor(hparams)
    preprocessor.train_tokenizer(base_dir.joinpath("spm"))
    preprocessor.segment_and_write()
