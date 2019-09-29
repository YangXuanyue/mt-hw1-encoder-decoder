import configs
from model_utils import *
from typing import *
from model import Model
import os
from log import log
import time
import data_utils
import re
from nltk.translate.bleu_score import corpus_bleu
import json
from informative_lr_scheduler import InformativeLrScheduler


class Runner:
    def __init__(self):
        self.model = Model()
        self.xe_loss = nn.CrossEntropyLoss(ignore_index=data_utils.tgt_vocab.padding_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs.lr)
        self.lr_scheduler = InformativeLrScheduler(
            self.optimizer, 'min',
            patience=configs.lr_scheduler_patience,
            factor=configs.lr_scheduler_factor, verbose=True
        )
        self.epoch_idx = 0
        self.min_validating_ppl = 1000.
        self.ckpt_id = '0.0'
        self.prev_ckpt_path = ''

    def train(self):
        if configs.ckpt_id or configs.loads_ckpt or configs.loads_best_ckpt:
            self.load_ckpt()

        while self.epoch_idx < configs.epoch_num:
            log(f'starting epoch {self.epoch_idx}')
            log('training')

            avg_loss = 0.
            batch_num = 0
            next_logging_pct = .5

            start_time = time.time()

            for pct, batch in data_utils.gen_batches('train'):
                batch_num += 1
                self.model.train()
                (
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch,
                    # [max_tgt_len, batch_size], [batch_size]
                    tgt_sent_batch, tgt_len_batch,
                    _

                ) = batch

                self.optimizer.zero_grad()

                # [max_tgt_len - 1, batch_size, tgt_vocab_size]
                tgt_word_logit_vecs_batch = self.model(
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch,
                    # [max_tgt_len, batch_size], [batch_size]
                    tgt_sent_batch, tgt_len_batch,
                )

                loss = self.xe_loss(
                    # [(max_tgt_len - 1 * batch_size), tgt_vocab_size]
                    tgt_word_logit_vecs_batch.view(-1, data_utils.tgt_vocab.size),
                    # [(max_tgt_len - 1) * batch_size]
                    tgt_sent_batch[1:].contiguous().view(-1).to(torch.device(configs.decoder_device_id))
                )

                loss.backward()

                if configs.clips_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=configs.max_grad_norm)

                self.optimizer.step()
                avg_loss += loss.item()

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, avg_train_loss: {avg_loss / batch_num:.6}, '
                        f'avg_train_ppl: {math.exp(avg_loss / batch_num):.6}, '
                        f'time: {time.time() - start_time:.6}'
                    )
                    next_logging_pct += 10.

            log(
                f'100%, avg_train_loss: {avg_loss / batch_num:.6}, '
                f'avg_train_ppl: {math.exp(avg_loss / batch_num):.6}, '
                f'time: {time.time() - start_time:.6}'
            )

            self.validate()
            self.epoch_idx += 1

    def validate(self):
        with torch.no_grad():
            log('validating')

            self.model.eval()
            batch_num = 0
            avg_loss = 0.
            next_logging_pct = .5

            start_time = time.time()

            for pct, batch in data_utils.gen_batches('valid'):
                batch_num += 1
                (
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch,
                    # [max_tgt_len, batch_size], [batch_size]
                    tgt_sent_batch, tgt_len_batch,
                    _
                ) = batch

                # [max_tgt_len - 1, batch_size, tgt_vocab_size]
                tgt_word_logit_vecs_batch = self.model(
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch,
                    # [max_tgt_len, batch_size], [batch_size]
                    tgt_sent_batch, tgt_len_batch,
                )

                loss = self.xe_loss(
                    # [(max_tgt_len - 1 * batch_size), tgt_vocab_size]
                    tgt_word_logit_vecs_batch.view(-1, data_utils.tgt_vocab.size),
                    # [(max_tgt_len - 1) * batch_size]
                    tgt_sent_batch[1:, :].contiguous().view(-1).to(torch.device(configs.decoder_device_id))
                )
                avg_loss += loss.item()

                # avg_dist += self.calc_avg_dist(best_seq_batch, transcript_batch, transcript_len_batch)

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, avg_dev_loss: {avg_loss / batch_num:.6}, '
                        f'avg_dev_ppl: {math.exp(avg_loss / batch_num):.6}, '
                        f'time: {time.time() - start_time:.6}'
                    )
                    next_logging_pct += 10.

            avg_loss /= batch_num
            avg_ppl = math.exp(avg_loss)

            log(
                f'{int(pct)}%, avg_dev_loss: {avg_loss:.6}, '
                f'avg_dev_ppl: {avg_ppl:.6}, '
                f'time: {time.time() - start_time:.6}'
            )

            if self.lr_scheduler.step(avg_loss) and configs.training and configs.backtracks_when_worse:
                configs.uses_new_optimizer = True
                self.load_ckpt(self.prev_ckpt_path)
                self.epoch_idx -= 1

            saved_ckpt = False

            if avg_ppl < self.min_validating_ppl:
                self.min_validating_ppl = avg_ppl

                min_validating_ppl_file = open(configs.min_validating_ppl_path)

                if avg_ppl < float(min_validating_ppl_file.readline().strip()):
                    min_validating_ppl_file.close()
                    min_validating_ppl_file = open(configs.min_validating_ppl_path, 'w')
                    print(avg_ppl, file=min_validating_ppl_file)
                    self.save_ckpt()
                    saved_ckpt = True

                min_validating_ppl_file.close()

            if configs.saves_every_ckpts and not saved_ckpt:
                self.save_ckpt()

    def test(self, name='test'):
        with torch.no_grad():
            log('testing')
            self.load_ckpt()
            self.model.eval()
            next_logging_pct = .5
            best_seqs = [[] for idx in range(data_utils.get_dataset_size(name))]
            start_time = time.time()

            for pct, batch in data_utils.gen_batches(name):
                (
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch,
                    _, _,
                    # [batch_size]
                    idx_batch
                ) = batch
                max_src_len, *_ = src_sent_batch.shape
                best_seq_batch = self.model.decode(
                    # [max_src_len, batch_size], [batch_size]
                    src_sent_batch, src_len_batch
                )

                for idx, best_seq in zip(idx_batch, best_seq_batch):
                    best_seqs[idx] = data_utils.tgt_vocab.to_words(best_seq)

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, time: {time.time() - start_time:.6}'
                    )
                    next_logging_pct += 10.

            references = data_utils.get_references(name)

            bleu_score = Runner.compute_bleu_score(
                references=references,
                hypotheses=best_seqs
            )

            log(
                f'100%, time: {time.time() - start_time:.6}, bleu: {bleu_score}'
            )

            with open(f'{name}_results.{self.ckpt_id}.txt', 'w') as results_file:
                # for best_seq, reference in zip(best_seqs, references):
                for best_seq in best_seqs:
                    print(' '.join(best_seq), file=results_file)
                    # print(' '.join(reference), file=results_file)
                    # print('', file=results_file)
            os.system(f'./multi-bleu.perl data/{name}.de-en.en < {name}_results.{self.ckpt_id}.txt')

    @staticmethod
    def compute_bleu_score(references: List[List[str]], hypotheses: List[List[str]]) -> float:
        if references[0][0] == '<s>':
            references = [r[1:-1] for r in references]

        return corpus_bleu(
            [[r] for r in references],
            hypotheses
        )

    def get_ckpt(self):
        return {
            'id': f'{configs.timestamp}.{self.epoch_idx}',
            'parent_id': self.ckpt_id,
            'epoch_idx': self.epoch_idx,
            'min_validating_ppl': self.min_validating_ppl,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

    def set_ckpt(self, ckpt_dict):
        self.ckpt_id = ckpt_dict['id']
        self.epoch_idx = ckpt_dict['epoch_idx'] + 1
        self.min_validating_ppl = ckpt_dict['min_validating_ppl']

        model_state_dict = self.model.state_dict()
        model_state_dict.update(
            {
                name: param
                for name, param in ckpt_dict['model'].items()
                if name in model_state_dict
            }
        )

        self.model.load_state_dict(model_state_dict)
        del model_state_dict

        if not configs.uses_new_optimizer:
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])

        del ckpt_dict

        torch.cuda.empty_cache()

    ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        ckpt_path = f'{configs.ckpts_dir}/{configs.timestamp}.{self.epoch_idx}.ckpt'
        log(f'saving checkpoint {ckpt_path}')
        torch.save(self.ckpt, f=ckpt_path)
        self.prev_ckpt_path = ckpt_path

    @staticmethod
    def to_timestamp_and_epoch_idx(ckpt_path_):
        date, time, epoch_idx = map(int, re.split(r'[-.]', ckpt_path_[:ckpt_path_.find('.ckpt')]))
        return date, time, epoch_idx

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if configs.ckpt_id:
                ckpt_path = f'{configs.ckpts_dir}/{configs.ckpt_id}.ckpt'
            elif configs.loads_best_ckpt:
                ckpt_path = configs.best_ckpt_path
            else:
                ckpt_paths = [path for path in os.listdir(f'{configs.ckpts_dir}/') if path.endswith('.ckpt')]
                ckpt_path = f'{configs.ckpts_dir}/{sorted(ckpt_paths, key=Runner.to_timestamp_and_epoch_idx)[-1]}'

        print(f'loading checkpoint {ckpt_path}')

        self.ckpt = torch.load(ckpt_path)


if __name__ == '__main__':
    runner = Runner()

    if configs.training:
        runner.train()
    elif configs.validating:
        runner.validate()
    else:
        runner.test()
