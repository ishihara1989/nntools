import torch.nn as nn

__all__ = ['xavier_uniform', 'wn_xavier']

def xavier_uniform(layer, w_init_gain='linear'):
    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(w_init_gain))
    return layer

def wn_xavier(layer, w_init_gain='linear'):
    return nn.utils.weight_norm(xavier_uniform(layer, w_init_gain=w_init_gain))
