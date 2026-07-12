# AstroJEPA smoke test config
img_size = 128
patch_size = 16
in_chans = 3
n_embd = 192
n_head = 3
n_layer = 4
predictor_n_embd = 128
predictor_n_head = 4
predictor_n_layer = 2
predictor_num_queries = 8
bias = False
dropout = 0.0
use_cls_token = True

num_target_blocks = 2
min_target_block_size = 4
max_target_block_size = 16
context_scale = 2.0

use_vicreg = False

learning_rate = 1e-4
min_lr = 1e-5
weight_decay = 0.04
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

ema_momentum = 0.996
ema_warmup_iters = 100

batch_size = 2
gradient_accumulation_steps = 1
max_iters = 2
warmup_iters = 10
lr_decay_iters = 100
decay_lr = True

eval_interval = 100
eval_iters = 2
log_interval = 1

num_checkpoints = 0
checkpoint_schedule = "log"

out_dir = "logs/smoke_test"
device = "cuda"
dtype = "bfloat16"
compile = False
num_workers = 0
master_process = True
stream_hf = True
shuffle_buffer_size = 100
dataset_revision = "v2.0"
