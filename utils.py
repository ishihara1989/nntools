import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['xavier_uniform', 'wn_xavier', 'calc_kernel_minimum_variance', 'calc_cross_utterance_speaker_code', 'large_margin_cosine_loss', 'kl_normal']

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
    log_confidence = log_confidence - log_confidence.max()
    confidence = log_confidence.exp()
    s_dim, u_dim, f_dim, t_dim = speaker_code_sample.size()
    mask = torch.ones([u_dim, u_dim], device=speaker_code_sample.device) - torch.eye(u_dim, device=speaker_code_sample.device)
    speaker_code = torch.einsum('suft,sugt,ui->sug', confidence, speaker_code_sample, mask) / torch.einsum('suft,ui->suf', confidence, mask)
    return speaker_code.contiguous()

def large_margin_cosine_loss(feature, margin=0.05, s=4):
    # feature: SxUxF
    s_dim, u_dim, f_dim = feature.size()
    n = F.normalize(feature, dim=2)
    cossim = torch.einsum('suf,tvf->stuv', n, n)
    label = torch.eye(s_dim, device=feature.device)
    m = margin*label
    return F.cross_entropy(torch.exp(s*(cossim - m[:, :, None, None])), torch.arange(s_dim, device=feature.device)[:,None, None].expand(s_dim, u_dim, u_dim))

def kl_normal(x):
    mean = x.mean(dim=2, keepdim=True)
    diff = x-mean
    sigma = diff.matmul(diff.transpose(1,2))/x.size(2)
    kl = -sigma.logdet().sum() + sigma.diagonal(dim1=1, dim2=2).sum() + mean.pow(2).sum()
    return 0.5*(kl - x.size(1))

if __name__ == "__main__":
    x = torch.randn(1,3,10000)
    print(kl_normal(x))