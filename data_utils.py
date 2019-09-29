import configs
import os

from tqdm import tqdm
from model_utils import *
from collections import defaultdict, Counter
from itertools import chain
import time
import json
import Levenshtein
import csv
import bisect
from vocab import Vocab

src_lang, tgt_lang = 'de', 'en'
src_vocab, tgt_vocab = map(
    lambda lang: Vocab(f'{configs.data_dir}/vocab.{lang}.txt'),
    (src_lang, tgt_lang)
)

mode_to_names = {
    'train': ('train', 'valid'),
    'valid': ('valid',),
    'test': ('test', 'valid')
}

corpora = {
    name: tuple(
        np.load(f'{configs.data_dir}/sents.ids.{name}.{lang}.npy', allow_pickle=True)
        for lang in (src_lang, tgt_lang)
    )
    for name in mode_to_names[configs.mode]
}

name_to_references = {
    name: None
    for name in mode_to_names[configs.mode]
}


def get_references(name):
    if not name_to_references[name]:
        with open(f'{configs.data_dir}/{name}.{src_lang}-{tgt_lang}.{tgt_lang}') as corpus_file:
            name_to_references[name] = [
                sent
                for sent in map(lambda s: s.strip().split(' '), corpus_file.readlines())
            ]

    return name_to_references[name]


class Dataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        self.src_sents, self.tgt_sents = corpora[self.name]
        assert len(self.src_sents) == len(self.tgt_sents)

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        return self.src_sents[idx], self.tgt_sents[idx], idx


datasets = {
    name: Dataset(name)
    for name in mode_to_names[configs.mode]
}


def get_dataset_size(name):
    return len(datasets[name])


def collate_sent_batch(sent_batch):
    len_batch = [len(sent) for sent in sent_batch]
    max_len = max(len_batch)
    sent_batch = [
        np.concatenate(
            # padding_id = 0
            (sent, np.zeros(max_len - len(sent))),
            axis=0
        ) if len(sent) < max_len else sent
        for sent in sent_batch
    ]
    # [max_len, batch_size], [batch_size]
    return torch.LongTensor(sent_batch).t(), torch.LongTensor(len_batch)


def collate(batch):
    src_sent_batch, tgt_sent_batch, idx_batch = zip(*batch)
    src_sent_batch, src_len_batch = collate_sent_batch(src_sent_batch)
    tgt_sent_batch, tgt_len_batch = collate_sent_batch(tgt_sent_batch)
    return (
        # [max_src_len, batch_size], [batch_size]
        src_sent_batch.to(configs.encoder_device_id), src_len_batch,
        # [max_tgt_len, batch_size], [batch_size]
        tgt_sent_batch.to(configs.decoder_device_id), tgt_len_batch,
        idx_batch
    )


data_loaders = {
    name: tud.DataLoader(
        dataset=datasets[name],
        batch_size=configs.batch_size,
        shuffle=(name == 'train'),
        collate_fn=collate,
        num_workers=0
    )
    for name in mode_to_names[configs.mode]
}


def gen_batches(name):
    instance_num = 0

    for batch in data_loaders[name]:
        instance_num += len(batch[-1])
        pct = instance_num * 100. / len(datasets[name])
        yield pct, batch
