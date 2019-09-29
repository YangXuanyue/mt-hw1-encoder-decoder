import configs
from model_utils import *
from vocab import Vocab
import data_utils
from modules import *


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(vocab=data_utils.src_vocab).to(configs.encoder_device_id)
        self.decoder = Decoder(vocab=data_utils.tgt_vocab).to(configs.decoder_device_id)

        self.init_params()

    def init_params(self):
        self.apply(init_params_uniformly if configs.inits_params_uniformly else init_params)

    def forward(
        self,
        # [max_src_len, batch_size], [batch_size]
        src_sent_batch: torch.LongTensor, src_len_batch: torch.LongTensor,
        # [max_tgt_len, batch_size], [batch_size]
        tgt_sent_batch: torch.LongTensor, tgt_len_batch: torch.LongTensor
    ):
        # [max_len, batch_size, hidden_size], ([layer_num, direction_num, batch_size, hidden_size] * 2)
        src_states_batches, src_final_hidden_state_batches, src_final_cell_state_batches = map(
            lambda xs: [x.to(configs.decoder_device_id) for x in xs],
            self.encoder(
                src_sent_batch, src_len_batch
            )
        )

        tgt_word_logits_seq_batch = self.decoder(
            tgt_sent_batch, tgt_len_batch,
            src_states_batches, src_final_hidden_state_batches, src_final_cell_state_batches, src_len_batch
        )

        return tgt_word_logits_seq_batch

    def decode(
        self,
        # [max_src_len, batch_size], [batch_size]
        src_sent_batch, src_len_batch, max_len=200
    ):
        # [max_len, batch_size, hidden_size], ([layer_num, direction_num, batch_size, hidden_size] * 2)
        src_states_batches, src_final_hidden_state_batches, src_final_cell_state_batches = map(
            lambda xs: [x.to(configs.decoder_device_id) for x in xs],
            self.encoder(
                src_sent_batch, src_len_batch
            )
        )

        best_seq_batch = self.decoder.decode(
            src_states_batches, src_final_hidden_state_batches, src_final_cell_state_batches, src_len_batch,
            max_len=max_len
        )

        return best_seq_batch
