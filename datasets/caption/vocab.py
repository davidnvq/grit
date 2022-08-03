# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
from __future__ import unicode_literals
from collections import defaultdict
import logging
import os
import json

logger = logging.getLogger(__name__)


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(
        self,
        counter=None,
        max_size=None,
        min_freq=1,
        specials=['<pad>'],
        vocab_path=None,
    ):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vocab_path: path to the json file
        """
        if vocab_path is not None and os.path.exists(vocab_path):
            vocab_data = json.load(open(vocab_path))
            self.freqs = vocab_data['freqs']
            self.itos = vocab_data['itos']
            self.stoi = defaultdict(_default_unk_index)
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        else:
            self.freqs = counter
            counter = counter.copy()
            min_freq = max(min_freq, 1)

            self.itos = list(specials)
            # frequencies of special tokens are not counted when building vocabulary
            # in frequency order
            for tok in specials:
                del counter[tok]

            max_size = None if max_size is None else max_size + len(self.itos)

            # sort by frequency, then alphabetically
            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

            for word, freq in words_and_frequencies:
                if freq < min_freq or len(self.itos) == max_size:
                    break
                self.itos.append(word)

            self.stoi = defaultdict(_default_unk_index)
            # stoi is simply a reverse dict for itos
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        if not isinstance(v, list):
            words = sorted(v.itos) if sort else v.itos
        else:
            words = set(v)
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


def _default_unk_index():
    return 0