import configs
from model_utils import *
from vocab import Vocab


class BilinearAttention(nn.Module):
    def __init__(
        self
    ):
        super().__init__()


class ScaledDotProductAttention(nn.Module):
    # http://nlp.seas.harvard.edu/2018/04/03/attention.html
    def __init__(
        self,
        context_size,
        raw_query_size,
        hidden_size
    ):
        super().__init__()
        self.context_size, self.raw_query_size = context_size, raw_query_size
        self.hidden_size = hidden_size
        self.keys_batch, self.values_batch, self.masks_batch = None, None, None
        self.factor = configs.query_size ** -.5

        self.key_projector = nn.Linear(self.context_size, self.hidden_size)
        self.value_projector = nn.Linear(self.context_size, self.hidden_size)
        self.query_projector = nn.Linear(self.raw_query_size, self.hidden_size)

    def clear(self):
        self.keys_batch, self.values_batch, self.masks_batch = None, None, None

    def set(
        self,
        # [batch_size, seq_len, context_size]
        contexts_batch,
        # [batch_size, seq_len]
        masks_batch=None
    ):
        # [batch_size, seq_len, hidden_size]
        self.keys_batch = self.key_projector(contexts_batch)
        # [batch_size, seq_len, hidden_size]
        self.values_batch = self.value_projector(contexts_batch)
        # [batch_size, seq_len]
        self.masks_batch = masks_batch

    def append(
        self,
        # [batch_size, context_size]
        context_batch,
        # [batch_size]
        mask_batch=None
    ):
        batch_size, _ = context_batch.shape
        # [batch_size, 1, hidden_size]
        key_batch = self.key_projector(context_batch).view(batch_size, 1, -1)
        # [batch_size, 1, hidden_size]
        value_batch = self.value_projector(context_batch).view(batch_size, 1, -1)

        if mask_batch is not None:
            mask_batch = mask_batch.view(batch_size, 1)

        if self.keys_batch is None:
            self.keys_batch, self.values_batch, self.masks_batch = key_batch, value_batch, mask_batch
        else:
            # [batch_size, seq_len, hidden_size]
            self.keys_batch = torch.cat(
                (self.keys_batch, key_batch),
                dim=1
            )
            # [batch_size, seq_len, hidden_size]
            self.values_batch = torch.cat(
                (self.values_batch, value_batch),
                dim=1
            )

            if mask_batch is not None:
                # [batch_size, seq_len]
                self.masks_batch = torch.cat(
                    (self.masks_batch, mask_batch),
                    dim=1
                )

    # a workaround for reindexing in beam search
    def __getitem__(
        self,
        # [batch_size]
        idx_batch
    ):
        if self.keys_batch is not None:
            # [batch_size, seq_len, hidden_size]
            self.keys_batch = self.keys_batch[idx_batch]
            # [batch_size, seq_len, hidden_size]
            self.values_batch = self.values_batch[idx_batch]
            # [batch_size, seq_len]
            self.masks_batch = self.masks_batch[idx_batch]

        return self

    def get_context_batch(
        self,
        # [batch_size, raw_query_size]
        raw_query_batch,
    ):
        # [batch_size, hidden_size]
        return self(
            # [1, batch_size, raw_query_size]
            raw_query_batch.view(1, *raw_query_batch.shape)
        ).view(-1, self.hidden_size)

    def forward(
        self,
        # [query_seq_len, batch_size, raw_query_size]
        raw_queries_batch
    ):
        # [batch_size, query_seq_len, raw_query_size]
        raw_queries_batch = raw_queries_batch.transpose(0, 1)
        batch_size, query_seq_len, raw_query_size = raw_queries_batch.shape
        # [batch_size, query_seq_len, 1, 1, hidden_size]
        queries_batch = self.query_projector(raw_queries_batch).view(batch_size, query_seq_len, 1, 1, self.hidden_size)
        _, key_seq_len, _ = self.keys_batch.shape
        # [batch_size, 1, key_seq_len, hidden_size, 1]
        keys_batch = self.keys_batch.view(batch_size, 1, key_seq_len, self.hidden_size, 1)
        # [batch_size, query_seq_len, key_seq_len]
        # = ([batch_size, query_seq_len, 1, 1, hidden_size]
        #    @ [batch_size, 1, key_seq_len, hidden_size, 1]).view(batch_size, query_seq_len, key_seq_len)
        pairwise_score_mat_batch = (queries_batch @ keys_batch).view(batch_size, query_seq_len, key_seq_len)

        if self.masks_batch is not None:
            # [batch_size, query_seq_len, key_seq_len]
            pairwise_score_mat_batch += torch.log(
                self.masks_batch.view(batch_size, 1, key_seq_len).float().to(pairwise_score_mat_batch.device)
            )

        # [batch_size, query_seq_len, key_seq_len]
        pairwise_score_mat_batch = F.softmax(
            pairwise_score_mat_batch * self.factor,
            dim=-1
        )
        # [batch_size, query_seq_len, hidden_size]
        # = ([batch_size, query_seq_len, key_seq_len] @ [batch_size, key_seq_len, hidden_size])
        return F.dropout(pairwise_score_mat_batch @ self.values_batch, p=configs.dropout_prob, training=self.training)


# zoneout
# https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/recurrent.py


class WeightDroppedRnn(nn.Module):
    def __init__(
        self, rnn,
        dropout_prob=configs.rnn_weights_dropout_prob
    ):
        super().__init__()
        self.rnn = rnn
        self.weight_names = [name for name, param in self.rnn.named_parameters() if 'weight_hh' in name]
        self.dropout_prob = dropout_prob
        self.rnn.flatten_parameters = self.flatten_parameters

        for name in self.weight_names:
            param = getattr(self.rnn, name)
            del self.rnn._parameters[name]
            self.rnn.register_parameter(name + '_raw', nn.Parameter(param.data))

    def flatten_parameters(self):
        return

    def forward(self, *args, **kwargs):
        for name in self.weight_names:
            raw_param = getattr(self.rnn, name + '_raw')
            param_data = F.dropout(raw_param.data, p=self.dropout_prob, training=self.training)
            setattr(self.rnn, name, nn.Parameter(param_data))

        return self.rnn(*args, **kwargs)


class WeightDroppedRnnCell(nn.Module):
    def __init__(self, rnn_cell, dropout_prob=configs.rnn_weights_dropout_prob):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.weight_names = [name for name, param in self.rnn_cell.named_parameters() if 'weight_hh' in name]
        self.dropout_prob = dropout_prob
        self.rnn_cell.flatten_parameters = self.flatten_parameters

        for name in self.weight_names:
            param = getattr(self.rnn_cell, name)
            del self.rnn_cell._parameters[name]
            self.rnn_cell.register_parameter(name + '_raw', nn.Parameter(param.data))

    def flatten_parameters(self):
        return

    def start(self):
        for name in self.weight_names:
            raw_param = getattr(self.rnn_cell, name + '_raw')
            param_data = F.dropout(raw_param.data, p=self.dropout_prob, training=self.training)
            setattr(self.rnn_cell, name, nn.Parameter(param_data))

    def forward(self, *args, **kwargs):
        return self.rnn_cell(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.embedder = nn.Embedding(
            embedding_dim=configs.embedding_dim,
            num_embeddings=self.vocab.size,
            padding_idx=self.vocab.padding_id
        )
        self.direction_num = 2
        self.rnns = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=(configs.embedding_dim if i == 0 else configs.rnn_hidden_size),
                    hidden_size=(configs.rnn_hidden_size // self.direction_num),
                    # num_layers=configs.rnn_layer_num,
                    bidirectional=True,
                )
                for i in range(configs.rnn_layer_num)
            ]
        )

        if configs.uses_weight_dropped_rnn:
            for i in range(configs.rnn_layer_num):
                self.rnns[i] = WeightDroppedRnn(self.rnns[i])

    def forward(
        self,
        # [max_src_len, batch_size], [batch_size]
        src_sent_batch, src_len_batch
    ):
        batch_size, = src_len_batch.shape

        sorted_idx_batch = sorted(range(batch_size), key=src_len_batch.__getitem__, reverse=True)
        orig_idx_batch = [-1] * batch_size

        for sorted_idx, orig_idx in enumerate(sorted_idx_batch):
            orig_idx_batch[orig_idx] = sorted_idx

        src_sent_batch = src_sent_batch[:, sorted_idx_batch]
        src_len_batch = src_len_batch[sorted_idx_batch]

        # [max_src_len, batch_size, embedding_dim]
        src_embeddings_batch = F.dropout(self.embedder(src_sent_batch), p=configs.dropout_prob, training=self.training)
        src_embeddings_batch = rnn_utils.pack_padded_sequence(src_embeddings_batch, src_len_batch)

        hidden_states_batches, final_hidden_state_batches, final_cell_state_batches = [], [], []

        for i in range(configs.rnn_layer_num):
            # [max_src_len, batch_size, hidden_size * 2], ([layer_num * 2, batch_size, hidden_size / 2] * 2)
            hidden_states_batch, (final_hidden_state_batch, final_cell_state_batch) = self.rnns[i](
                src_embeddings_batch if i == 0 else hidden_states_batches[i - 1]
            )
            hidden_states_batches.append(hidden_states_batch)
            final_hidden_state_batches.append(final_hidden_state_batch)
            final_cell_state_batches.append(final_cell_state_batch)

        for i in range(configs.rnn_layer_num):
            hidden_states_batches[i], _ = rnn_utils.pad_packed_sequence(hidden_states_batches[i])

            hidden_states_batches[i] = hidden_states_batches[i][:, orig_idx_batch]
            # ([direction_num, batch_size, hidden_size / 2] * 2)
            final_hidden_state_batches[i], final_cell_state_batches[i] = map(
                lambda x: x[:, orig_idx_batch].view(
                    self.direction_num, batch_size, configs.rnn_hidden_size // self.direction_num
                ),
                (final_hidden_state_batches[i], final_cell_state_batches[i])
            )
            # ([batch_size, hidden_size] * 2)
            final_hidden_state_batches[i], final_cell_state_batches[i] = map(
                lambda x: x.transpose(0, 1).contiguous().view(batch_size, configs.rnn_hidden_size),
                (final_hidden_state_batches[i], final_cell_state_batches[i])
            )

            if configs.detaches_src_final_states:
                final_hidden_state_batches[i], final_cell_state_batches[i] = map(
                    lambda x: x.detach(),
                    (final_hidden_state_batches[i], final_cell_state_batches[i])
                )

        return (
            # [[max_src_len, batch_size, hidden_size] * layer_num], ([[batch_size, hidden_size] * layer_num] * 2)
            hidden_states_batches, final_hidden_state_batches, final_cell_state_batches
        )


class Decoder(nn.Module):
    def __init__(self, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.embedder = nn.Embedding(
            embedding_dim=configs.embedding_dim,
            num_embeddings=self.vocab.size,
            padding_idx=self.vocab.padding_id
        )
        self.rnn_cells = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_size=(configs.embedding_dim + configs.rnn_hidden_size if i == 0
                                else configs.rnn_hidden_size * 2),
                    hidden_size=configs.rnn_hidden_size
                )
                for i in range(configs.rnn_layer_num)
            ]
        )

        if configs.uses_weight_dropped_rnn:
            for i in range(configs.rnn_layer_num):
                self.rnn_cells[i] = WeightDroppedRnnCell(self.rnn_cells[i])

        self.src_attentions = nn.ModuleList(
            [
                ScaledDotProductAttention(
                    context_size=configs.rnn_hidden_size,
                    raw_query_size=configs.rnn_hidden_size,
                    hidden_size=configs.rnn_hidden_size
                )
                for i in range(configs.rnn_layer_num)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                configs.rnn_hidden_size * 2,
                configs.rnn_hidden_size
            ),
            nn.Tanh(),
            nn.Dropout(p=configs.dropout_prob),
            nn.Linear(
                configs.rnn_hidden_size,
                self.vocab.size
            )
        )

        if configs.ties_weight:
            self.classifier[3].weight = self.embedder.weight

    def forward(
        self,
        # [max_tgt_len, batch_size]
        tgt_sent_batch,
        # [batch_size]
        tgt_len_batch,
        # [max_src_len, batch_size, hidden_size * 2]
        src_states_batches,
        # ([layer_num, batch_size, hidden_size * 2] * 2)
        src_final_hidden_state_batches, src_final_cell_state_batches,
        # [batch_size]
        src_len_batch
    ):
        # [batch_size, max_src_len, hidden_size * 2]
        for i in range(configs.rnn_layer_num):
            src_states_batches[i] = src_states_batches[i].transpose(0, 1)

        max_tgt_len, batch_size = tgt_sent_batch.shape
        # [seq_len, batch_size, embedding_dim]
        tgt_embeddings_batch = F.dropout(self.embedder(tgt_sent_batch), p=configs.dropout_prob, training=self.training)

        _, max_src_len, _ = src_states_batches[0].shape
        # [batch_size, max_src_len]
        src_masks_batch = build_len_masks_batch(src_len_batch, max_src_len)

        for i in range(configs.rnn_layer_num):
            self.src_attentions[i].set(
                # [batch_size, max_src_len, context_size]
                contexts_batch=src_states_batches[i],
                # [batch_size, max_src_len]
                masks_batch=src_masks_batch
            )

        next_embedding_batch = tgt_embeddings_batch[0]
        next_word_logit_vecs_batch = []

        if configs.uses_weight_dropped_rnn:
            for i in range(configs.rnn_layer_num):
                self.rnn_cells[i].start()

        # ([batch_size, hidden_size] * 2)
        # hidden_state_batches, cell_state_batches = map(
        #     lambda x: [x[i] for i in range(configs.rnn_layer_num)],
        #     (src_final_hidden_state_batch, src_final_cell_state_batch,)
        # )
        hidden_state_batches, cell_state_batches = src_final_hidden_state_batches, src_final_cell_state_batches

        prev_context_batches = [
            torch.zeros(
                (batch_size, configs.rnn_hidden_size)
            ).float().to(configs.decoder_device_id)
            for i in range(configs.rnn_layer_num)
        ]

        for t in range(max_tgt_len - 1):
            for i in range(configs.rnn_layer_num):
                # [batch_size, hidden_size], [batch_size, hidden_size]
                hidden_state_batches[i], cell_state_batches[i] = self.rnn_cells[i](
                    # [batch_size, embedding_dim + value_size]
                    torch.cat(
                        (
                            # [batch_size, hidden_size]
                            next_embedding_batch if i == 0 else hidden_state_batches[i - 1],
                            # [batch_size, value_size]
                            prev_context_batches[i]

                        ), dim=-1
                    ),
                    (hidden_state_batches[i], cell_state_batches[i])
                )

                # [batch_size, value_size]
                prev_context_batches[i] = self.src_attentions[i].get_context_batch(
                    raw_query_batch=hidden_state_batches[i]
                )

            # [batch_size, vocab_size]
            next_word_logit_vec_batch = self.classifier(
                torch.cat(
                    (
                        # [batch_size, embedding_dim]
                        hidden_state_batches[-1],
                        # [batch_size, value_size]
                        prev_context_batches[-1]
                    ), dim=-1
                )
            )
            next_word_logit_vecs_batch.append(next_word_logit_vec_batch)

            if self.training and random.random() < configs.sampling_rate:
                # [batch_size, vocab_size]
                next_word_prob_vec_batch = F.gumbel_softmax(next_word_logit_vec_batch, tau=.5, hard=True)

                if configs.uses_gumbel_softmax:
                    # [batch_size, embedding] = [batch_size, vocab_size] @ [vocab_size, embedding_dim]
                    next_embedding_batch = (next_word_prob_vec_batch @ self.embedder.weight)
                else:
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(
                        # [batch_size]
                        torch.argmax(next_word_prob_vec_batch, dim=-1)
                    )

                next_embedding_batch = F.dropout(next_embedding_batch, p=configs.dropout_prob, training=self.training)
            else:
                next_embedding_batch = tgt_embeddings_batch[t + 1]

        # [max_tgt_len - 1, batch_size, vocab_size]
        next_word_logit_vecs_batch = torch.stack(
            # (max_tgt_len - 1) * [batch_size, vocab_size]
            next_word_logit_vecs_batch,
            dim=0
        )

        return next_word_logit_vecs_batch

    def decode(
        self,
        src_states_batches,
        # ([layer_num, batch_size, hidden_size * 2] * 2)
        src_final_hidden_state_batches, src_final_cell_state_batches,
        # [batch_size]
        src_len_batch,
        beam_width=configs.beam_width,
        per_node_beam_width=configs.per_node_beam_width,
        max_len=200
    ):
        assert not self.training

        batch_size, = src_len_batch.shape
        # [batch_size, max_encoded_utterance, hidden_size]
        # [batch_size, max_src_len, hidden_size * 2]
        for i in range(configs.rnn_layer_num):
            src_states_batches[i] = src_states_batches[i].transpose(0, 1)

        best_seq_batch = []
        # [vocab_size]
        end_word_prob_vec = torch.zeros((self.vocab.size,), dtype=torch.float32)
        end_word_prob_vec[self.vocab.end_id] = 1.
        # [vocab_size]
        end_word_log_prob_vec = torch.log(end_word_prob_vec).to(configs.decoder_device_id)
        # [beam_width, vocab_size]
        end_word_log_prob_vec_beam = end_word_log_prob_vec.view(1, -1).repeat(beam_width, 1).to(
            configs.decoder_device_id)

        # ([batch_size, hidden_size * 2] * 2)
        hidden_state_batches, cell_state_batches = src_final_hidden_state_batches, src_final_cell_state_batches

        for idx_in_batch in range(batch_size):
            src_len = src_len_batch[idx_in_batch]

            for i in range(configs.rnn_layer_num):
                self.src_attentions[i].clear()
                self.src_attentions[i].set(
                    # [beam_width, src_len, hidden_size]
                    contexts_batch=src_states_batches[i][idx_in_batch, :src_len].repeat(beam_width, 1, 1)
                )

            # ([[beam_width, hidden_size] * layer_num] * 2)
            hidden_state_beams, cell_state_beams = map(
                lambda xs: [
                    x[idx_in_batch].repeat(beam_width, 1)
                    for x in xs
                ],
                (hidden_state_batches, cell_state_batches)
            )

            if configs.uses_weight_dropped_rnn:
                for i in range(configs.rnn_layer_num):
                    self.rnn_cells[i].start()

            # [beam_width]
            next_word_id_beam = torch.full(
                (beam_width,), self.vocab.start_id, dtype=torch.long
            ).to(configs.decoder_device_id)
            # [beam_width, embedding_dim]
            next_embedding_beam = self.embedder(next_word_id_beam.to(configs.decoder_device_id))
            # [beam_width, seq_len(just 1 now)]
            word_id_seq_beam = next_word_id_beam.view(beam_width, 1)
            # [beam_width]
            word_id_seq_len_beam = torch.full((beam_width,), 0, dtype=torch.long).to(configs.decoder_device_id)
            # [beam_width]
            seq_log_prob_beam = torch.zeros(beam_width).to(configs.decoder_device_id)

            prev_context_beams = [
                torch.zeros(
                    (beam_width, configs.rnn_hidden_size)
                ).float().to(configs.decoder_device_id)
                for i in range(configs.rnn_layer_num)
            ]

            for t in range(max_len):
                end_flag_beam = next_word_id_beam.eq(self.vocab.end_id)

                for i in range(configs.rnn_layer_num):
                    # [batch_size, hidden_size], [batch_size, hidden_size]
                    hidden_state_beams[i], cell_state_beams[i] = self.rnn_cells[i](
                        # [batch_size, embedding_dim + value_size]
                        torch.cat(
                            (
                                # [batch_size, hidden_size]
                                next_embedding_beam if i == 0 else hidden_state_beams[i - 1],
                                # [batch_size, value_size]
                                prev_context_beams[i]

                            ), dim=-1
                        ),
                        (hidden_state_beams[i], cell_state_beams[i])
                    )

                    # [batch_size, value_size]
                    prev_context_beams[i] = self.src_attentions[i].get_context_batch(
                        raw_query_batch=hidden_state_beams[i]
                    )

                # [beam_width, vocab_size]
                next_word_logit_vec_beam = self.classifier(
                    torch.cat(
                        (
                            hidden_state_beams[-1],
                            prev_context_beams[-1]

                        ), dim=-1
                    )
                )
                # [beam_width, vocab_size]
                end_flag_vec_beam = end_flag_beam.view(-1, 1).repeat(1, self.vocab.size).to(configs.decoder_device_id)
                # [beam_width, vocab_size]
                next_word_log_prob_vec_beam = torch.where(
                    # [beam_width, vocab_size]
                    end_flag_vec_beam,
                    # [beam_width, vocab_size]
                    end_word_log_prob_vec_beam,
                    # [beam_width, vocab_size]
                    F.log_softmax(next_word_logit_vec_beam, dim=-1)
                )
                # [beam_width, vocab_size] = [beam_width, 1] + [beam_width, vocab_size]
                next_word_log_prob_vec_beam = seq_log_prob_beam.view(-1, 1) + next_word_log_prob_vec_beam
                # [beam_width, vocab_size]
                word_id_seq_len_vec_beam = word_id_seq_len_beam.view(-1, 1).repeat(1, self.vocab.size)
                # [beam_width, vocab_size]
                word_id_seq_len_vec_beam[~end_flag_vec_beam] += 1
                # [beam_width, vocab_size] = [beam_width, vocab_size] / [beam_width, vocab_size]
                normalized_next_word_log_prob_vec_beam = next_word_log_prob_vec_beam / word_id_seq_len_vec_beam.float()
                # [beam_width, per_node_beam_width], [beam_width, per_node_beam_width]
                top_normalized_next_word_log_prob_vec_beam, top_next_word_id_vec_beam = torch.topk(
                    # [beam_width, vocab_size] -> # [beam_width, (top)per_node_beam_width]
                    normalized_next_word_log_prob_vec_beam, per_node_beam_width
                )
                # [beam_width * per_node_beam_width]
                all_top_normalized_next_word_log_prob_vec = top_normalized_next_word_log_prob_vec_beam.view(-1)
                # [beam_width * per_node_beam_width]
                all_top_next_word_id_vec = top_next_word_id_vec_beam.view(-1)
                # [beam_width], [beam_width]
                normalized_next_word_log_prob_beam, idx_beam = torch.topk(
                    # [beam_width * per_node_beam_width] -> [(top)beam_width]
                    all_top_normalized_next_word_log_prob_vec, beam_width
                )
                # idx = idx_in_beam * per_node_beam_width + idx_in_next_beam
                # [beam_width]
                prev_idx_beam = idx_beam / per_node_beam_width
                # [beam_width]
                next_word_id_beam = all_top_next_word_id_vec[idx_beam]
                next_embedding_beam = self.embedder(next_word_id_beam)

                for i in range(configs.rnn_layer_num):
                    hidden_state_beams[i] = hidden_state_beams[i][prev_idx_beam]
                    # [beam_width, hidden_size]
                    cell_state_beams[i] = cell_state_beams[i][prev_idx_beam]
                    prev_context_beams[i] = prev_context_beams[i][prev_idx_beam]

                # [beam_width, seq_len - 1]
                word_id_seq_beam = word_id_seq_beam[prev_idx_beam]

                # [beam_width, seq_len]
                word_id_seq_beam = torch.cat(
                    (
                        # [beam_width, seq_len - 1]
                        word_id_seq_beam,
                        # [beam_width, 1]
                        next_word_id_beam.view(beam_width, 1)
                    ), dim=1
                )
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                seq_log_prob_beam = next_word_log_prob_vec_beam[prev_idx_beam, next_word_id_beam]
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                word_id_seq_len_beam = word_id_seq_len_vec_beam[prev_idx_beam, next_word_id_beam]

                if next_word_id_beam.eq(self.vocab.end_id).all() or t == max_len - 1:
                    # [<s>, ... (</s>)]
                    best_seq = word_id_seq_beam[0].cpu().numpy().tolist()
                    end_idx = best_seq.index(self.vocab.end_id) if self.vocab.end_id in best_seq else len(best_seq)
                    best_seq = best_seq[1:end_idx]
                    best_seq_batch.append(best_seq)

                    break

        return best_seq_batch


class Reshaper(nn.Module):
    def __init__(self, output_shape):
        super().__init__()

        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(*self.output_shape)


class Scaler(nn.Module):
    def __init__(self, alpha=16.):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha).cuda())

    def forward(self, input):
        return self.alpha * input


class Normalizer(nn.Module):
    def __init__(self, target_norm=1.):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, input: torch.Tensor):
        return input * self.target_norm / input.norm(p=2, dim=1, keepdim=True)
