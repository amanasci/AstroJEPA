# AstroJEPA ~1M parameter model config
# Lightweight model for quick experiments and debugging

# Data
dataset = "Smith42/galaxies"
dataset_revision = "v2.0"
split = "train"
stream_hf = True
shuffle_buffer_size = 1000

# Model
img_size = 512
patch_size = 16
in_chans = 3
n_embd = 128
n_head = 2
n_layer = 3
predictor_n_embd = 64
predictor_n_head = 2
predictor_n_layer = 2
predictor_num_queries = 4
bias = False
dropout = 0.0
use_cls_token = True

# Masking
num_target_blocks = 2
min_target_block_size = 4
max_target_block_size = 16
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
ema_warmup_iters = 1000

# Training
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 50000
warmup_iters = 1000
lr_decay_iters = 50000
decay_lr = True

# Eval
eval_interval = 2500
eval_iters = 50
log_interval = 50

# Checkpoint
num_checkpoints = 3
checkpoint_schedule = "log"
checkpoint_interval = 0

# System
out_dir = "logs/astrojepa001M"
device = "cuda"
dtype = "bfloat16"
compile = True
num_workers = 4
master_process = True
