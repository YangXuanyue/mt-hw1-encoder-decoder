import configs
from vocab import Vocab
from collections import defaultdict
import numpy as np
from itertools import chain

names = ('train', 'valid', 'test')
raw_corpora = defaultdict(list)
src_lang, tgt_lang = 'de', 'en'

for name in names:
    suffix = '.wmixerprep' if name == 'train' else ''

    for lang in (src_lang, tgt_lang):
        with open(f'{configs.data_dir}/{name}.{src_lang}-{tgt_lang}.{lang}{suffix}') as corpus_file:
            for sent in map(lambda s: s.strip().split(' '), corpus_file.readlines()):
                if lang == tgt_lang:
                    sent = ['<s>'] + sent + ['</s>']

                raw_corpora[f'{name}.{lang}'].append(sent)

vocabs = {
    lang: Vocab.build(
        path=f'{configs.data_dir}/vocab.{lang}.txt',
        words=(
            word
            for sent in raw_corpora[f'train.{lang}']
            for word in sent
        )
    )
    for lang in (src_lang, tgt_lang)
}

for name in names:
    for lang in (src_lang, tgt_lang):
        np.save(
            f'{configs.data_dir}/sents.ids.{name}.{lang}.npy',
            [
                vocabs[lang].to_ids(sent)
                for sent in raw_corpora[f'{name}.{lang}']
            ]
        )
