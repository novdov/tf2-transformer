import argparse
import os
from pathlib import Path

from transformer.model import Transformer
from transformer.trainer import Trainer
from transformer.utils.hparams_utils import Hparams

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--mode", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_dir = Path(__file__).resolve().parent
    hparams = Hparams(base_dir.joinpath("hparams.json"))

    model = Transformer(
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        d_model=hparams.d_model,
        d_ff=hparams.d_ff,
        encoder_vocab_size=hparams.encoder_vocab_size,
        decoder_vocab_size=hparams.decoder_vocab_size,
        dropout_rate=hparams.dropout_rate,
    )

    trainer = Trainer(hparams, model)
    trainer.train_and_evaluate(
        train_steps=hparams.train_steps,
        eval_steps=hparams.eval_steps,
        eval_frequency=hparams.eval_frequency,
        checkpoint_dir=output_dir,
    )
