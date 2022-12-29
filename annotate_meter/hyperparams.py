import torch

# vocab
pad_idx = 1
sos_idx = 2

# architecture

hidden_dim = 512 #default 512
embed_dim = 256  # default 256
n_layers = 2 # default: 2
dropout = 0.3   # default: 2
batch_size = 32
num_epochs = 100

# training
max_lr = 1e-4
cycle_length = 3000

# generation
max_len = 20
max_len_translate = 30

# system
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

