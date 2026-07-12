# AstroJEPA ~300M parameter model config
# Trains on the Smith42/galaxies dataset (DESI Legacy Survey)

# Data
dataset = "Smith42/galaxies"
dataset_revision = "v2.0"
split = "train"
stream_hf = True
shuffle_buffer_size = 20000

# Model
img_size = 512
patch_size = 16
in_chans = 3
n_embd = 1024
n_head = 16
n_layer = 11
predictor_n_embd = 512
predictor_n_head = 8
predictor_n_layer = 5
predictor_num_queries = 4  # must equal num_target_blocks for per-block prediction
bias = False
dropout = 0.0
use_cls_token = True

# Masking
num_target_blocks = 4
min_target_block_size = 16
max_target_block_size = 64
context_scale = 2.0

# Loss
use_vicreg = False
vicreg_lambda = 1.0
vicreg_mu = 1.0
vicreg_nu = 0.1

# Optimizer
learning_rate = 1.5e-4
min_lr = 1.5e-5
weight_decay = 0.04
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# EMA
ema_momentum = 0.996
ema_warmup_iters = 2000

# Training
batch_size = 16
gradient_accumulation_steps = 4
max_iters = 100000
warmup_iters = 2000
lr_decay_iters = 100000
decay_lr = True

# Eval
eval_interval = 5000
eval_iters = 100
log_interval = 100

# Checkpoint
num_checkpoints = 5
checkpoint_schedule = "log"
checkpoint_interval = 0

# System
out_dir = "logs/astrojepa300M"
device = "cuda"
dtype = "bfloat16"
compile = True
num_workers = 8
master_process = True
