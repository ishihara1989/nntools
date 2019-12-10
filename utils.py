import torch
import torch.nn as nn

__all__ = ['xavier_uniform', 'wn_xavier', 'calc_kernel_minimum_variance', 'calc_cross_utterance_speaker_code']

def xavier_uniform(layer, w_init_gain='linear'):
    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(w_init_gain))
    return layer

def wn_xavier(layer, w_init_gain='linear'):
    return nn.utils.weight_norm(xavier_uniform(layer, w_init_gain=w_init_gain))

def calc_kernel_minimum_variance(content_code):
    s_dim, u_dim, f_dim, t_dim = content_code.size()
    x = content_code.permute(0,1,3,2).contiguous().view([s_dim, u_dim*t_dim, f_dim])
    xx = x.pow(2).sum(dim=2) # S, T
    xy = x.matmul(x.transpose(1,2)) # S, T, T
    pdist = xx[:,:,None]+xx[:,None,:]-2*xy
    c = 1
    v = (c/(c+pdist)).sum()/(u_dim*t_dim)**2
    x = content_code.transpose(2,3).contiguous().view([s_dim*u_dim*t_dim, f_dim])
    xx = x.pow(2).sum(dim=1)
    xy = x.matmul(x.transpose(0,1))
    pdist = xx[:,None]+xx[None:]-2*xy
    v2 = (c/(c+pdist)).sum()/(u_dim*t_dim)**2/s_dim
    return v - v2

def calc_cross_utterance_speaker_code(log_confidence, speaker_code_sample):
    # log_confidence: SxUx1xT
    # speaker_code_sample: SxUxFxT
    # speaker_code: SxUxF
    confidence = log_confidence.exp()
    s_dim, u_dim, f_dim, t_dim = speaker_code_sample.size()
    mask = torch.ones([u_dim, u_dim], device=speaker_code_sample.device) - torch.eye(u_dim, device=speaker_code_sample.device)
    speaker_code = torch.einsum('suft,sugt,ui->sug', confidence, speaker_code_sample, mask) / torch.einsum('suft,ui->suf', confidence, mask)
    return speaker_code.contiguous()