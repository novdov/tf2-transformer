import json


class Hparams:

    def __init__(self, hparam_path):
        self._hparams = load_json(hparam_path)

    def __getattr__(self, key):
        try:
            return self._hparams[key]
        except KeyError:
            raise AttributeError(f"Cannot find {key}.")


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
