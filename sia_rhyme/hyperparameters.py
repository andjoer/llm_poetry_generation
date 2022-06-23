import torch

# vocab
pad_idx = 1
sos_idx = 2

# architecture

hidden_dim = 50 # default 512
embed_dim = 100  # default 256
lin_out_dim = 8
n_layers = 2     # default: 2
dropout = 0.25   # default: 2
batch_size = 64
num_epochs = 20

bidirectional = True

# training
max_lr = 1e-4
cycle_length = 3000

# generation
max_len = 20
max_len_translate = 30

# system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")