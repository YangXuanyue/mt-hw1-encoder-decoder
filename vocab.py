import configs
import os
# import numpy as np
import json
from collections import Counter, defaultdict
from itertools import chain
from model_utils import *
import time
from typing import *


class Vocab:
    @staticmethod
    def build(path, words: Iterable[str], cnt_cutoff=2, max_size=50000) -> 'Vocab':
        word_to_id, id_to_word = {}, []

        def add_word(word: str):
            if word in word_to_id: return
            word_to_id[word] = len(id_to_word)
            id_to_word.append(word)

        for word in ('<pad>', '<s>', '</s>', '<unk>', '<mask>'):
            add_word(word)

        word_cnts = Counter(words)
        top_words = sorted(
            (word for word, cnt in word_cnts.items() if cnt >= cnt_cutoff),
            key=word_cnts.__getitem__,
            reverse=True
        )[:max_size]

        for word in top_words:
            add_word(word)

        with open(path, 'w') as vocab_file:
            vocab_file.writelines('\n'.join(id_to_word))

        return Vocab(path)

    def __init__(self, path):
        assert os.path.exists(path)

        self.word_to_id, self.id_to_word = {}, []

        with open(path) as vocab_file:
            for word in map(lambda s: s.strip(), vocab_file.readlines()):
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)

        (
            self.padding_id,
            self.start_id,
            self.end_id,
            self.unk_id,
            self.mask_id
        ) = map(
            self.word_to_id.__getitem__,
            ('<pad>', '<s>', '</s>', '<unk>', '<mask>')
        )

        self.size = len(self.id_to_word)
        assert len(self.word_to_id) == self.size

    def __contains__(self, word: str):
        return word in self.word_to_id

    def __getitem__(self, word_or_id: Union[str, int]) -> Union[int, str]:
        if isinstance(word_or_id, str):
            return self.word_to_id.get(word_or_id, self.unk_id)
        else:
            return self.id_to_word[word_or_id]

    def __len__(self):
        return self.size

    def to_words(self, ids: Iterable[int]) -> List[str]:
        return list(map(self.id_to_word.__getitem__, ids))

    def to_ids(self, words: Iterable[str]) -> List[int]:
        return list(map(lambda word: self.word_to_id.get(word, self.unk_id), words))

