import datetime
import argparse as ap
from model_utils import *
import time
import random
import os
import shutil
import warnings

warnings.filterwarnings('ignore')

seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

arg_parser = ap.ArgumentParser()
arg_parser.add_argument('-m', '--mode', default='train')
arg_parser.add_argument('-c', '--ckpt', default=None)
# arg_parser.add_argument('-d', '--device', type=int, default=0)
arg_parser.add_argument('-b', action='store_true', default=False)
arg_parser.add_argument('-l', action='store_true', default=False)

args = arg_parser.parse_args()

mode = args.mode
training = args.mode == 'train'
validating = args.mode == 'dev'
# ckpt.best is trained with data_part_num = 3
ckpt_id = args.ckpt
# device_id = args.device
loads_best_ckpt = args.b
loads_ckpt = args.l

# training = True
# ckpt_id = None
# loads_best_ckpt = False
# loads_ckpt = False


timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')


class Dir:
    def __init__(self, name):
        self.name = name

        if not os.path.exists(name):
            os.mkdir(name)

    def __str__(self):
        return self.name


logs_dir = Dir('logs')
ckpts_dir = Dir('ckpts')
data_dir = Dir('data')
exps_dir = Dir('exps')

if training:
    for py_name in ('configs', 'modules', 'model', 'runner'):
        shutil.copyfile(f'{py_name}.py', f'{exps_dir}/{py_name}.{timestamp}.py')

min_validating_ppl_path = f'min_validating_ppl_path.{timestamp}.txt'

with open(min_validating_ppl_path, 'w') as min_validating_ppl_file:
    print(1000., file=min_validating_ppl_file)

best_ckpt_path = f'{ckpts_dir}/ckpt.best'

embedding_dim = 512
key_size, value_size, query_size = [256] * 3
# key_size, value_size, query_size = [128] * 3

lr = 1e-3
sets_new_lr = True
batch_size = 64
uses_bi_rnn = True
rnn_type = nn.LSTM
# rnn_type = 'gru'
rnn_hidden_size = 512
rnn_layer_num = 2
dropout_prob = .2
momentum = .9
l2_weight_decay = 5e-4
feature_num = 512

epoch_num = 240

# uses_center_loss = False
# uses_pairwise_sim_loss = False

normalizes_outputs = True
uses_new_classifier = False
uses_new_transformer = False
uses_new_encoder = False

uses_cnn = True

uses_batch_norm_encoder = False
uses_residual_rnns = False

loads_optimizer_state = not (sets_new_lr or uses_new_classifier or uses_new_encoder)

lr_scheduler_factor = .5
lr_scheduler_patience = 1

uses_weight_dropped_rnn = False

rnn_weights_dropout_prob = .2

sampling_rate = 0.2

encoder_device_id = 0
decoder_device_id = 0

per_node_beam_width = 32
beam_width = 32

uses_new_optimizer = True

uses_self_attention = True

uses_gumbel_softmax = True

uses_lang_model = False

use_multi_head_attention = True
supervises_encoder = False

clips_grad_norm = False
max_grad_norm = 5.

detaches_src_final_states = False
inits_params_uniformly = False
saves_every_ckpts = False
backtracks_when_worse = True
ties_weight = True

# uses_lm = not training and False
