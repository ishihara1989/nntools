import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

__all__ = ['Func', 'DenseResBlocks1d', 'InfusedResBlock1d', 'StackedInfusedResBlock1d', 'LinearResBlock', 'SelfAttention']

class Func(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, **self.kwargs)

    def __repr__(self):
        return 'Func(func={})'.format(self.func.__name__)

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.update = nn.Sequential(
            nn.ReflectionPad2d(1),
            wn_xavier(nn.Conv2d(channels, 2*channels, 3, bias=False)),
            Func(F.glu, dim=1),
            nn.ReflectionPad2d(1),
            wn_xavier(nn.Conv2d(channels, 2*channels, 3, bias=False)),
            Func(F.glu, dim=1)
        )

    def forward(self, x):
        r = self.update(x)
        return x + r

class DenseResBlocks1d(nn.Module):
    def __init__(self, n_channels, dilations):
        super(DenseResBlocks1d, self).__init__()
        self.n_layers = len(dilations)
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            self.convs.append(nn.Sequential(
                nn.ReplicationPad1d(dilation),
                utils.wn_xavier(nn.Conv1d(n_channels, 2*n_channels, 3, dilation=dilation)),
                Func(F.glu, dim=1)
            ))
            last_c = 4*n_channels if i < self.n_layers -1 else 2*n_channels
            self.skips.append(nn.Sequential(
                utils.wn_xavier(nn.Conv1d(n_channels, last_c, 1)),
                Func(F.glu, dim=1)
            ))

    # TODO: scale and bias is ignored??
    def forward(self, x, cond_scale=None, cond_bias=None):
        rs = x
        if cond_scale is not None:
            cond_scales = cond_scale.chunk(self.n_layers, dim=1)
        if cond_bias is not None:
            cond_biases = cond_bias.chunk(self.n_layers, dim=1)
        for i in range(self.n_layers):
            rs = self.convs[i](rs)
            if cond_scale is not None:
                rs = rs * cond_scales[i]
            if cond_bias is not None:
                rs = rs + cond_biases[i]
            rs = self.skips[i](rs)

            if i < self.n_layers-1:
                rs, skip = rs.chunk(2, dim=1)
                if i==0:
                    out = skip
                else:
                    out = out + skip
            else:
                out = out + rs
        return out

class InfusedResBlock1d(nn.Module):
    def __init__(self, base_dim):
        super(InfusedResBlock1d, self).__init__()
        self.update = nn.Sequential(
            utils.wn_xavier(nn.Conv1d(base_dim, 2*base_dim, 3, padding=1, bias=False)),
            Func(F.glu, dim=1),
            utils.wn_xavier(nn.Conv1d(base_dim, 2*base_dim, 3, padding=1, bias=False)),
            Func(F.glu, dim=1),
        )
        self.alpha = nn.Parameter(torch.ones(1, base_dim, 1))

    def forward(self, x, w, b):
        r = self.update(self.alpha*torch.sigmoid(w)*x+b)
        return x + r

class StackedInfusedResBlock1d(nn.Module):
    def __init__(self, base_dim, layers):
        super(StackedInfusedResBlock1d, self).__init__()
        self.layers = nn.ModuleList([InfusedResBlock1d(base_dim) for _ in range(layers)])
    
    def forward(self, x, cond):
        l = len(self.layers)
        w, b = torch.chunk(cond, 2, dim=1)
        ws = torch.chunk(w, l, dim=1)
        bs = torch.chunk(b, l, dim=1)
        r = x
        for i, layer in enumerate(self.layers):
            r = layer(r, ws[i][:,:,None], bs[i][:,:,None])
        return r


class LinearResBlock(nn.Module):
    def __init__(self, base_dim):
        super(LinearResBlock, self).__init__()
        self.update = nn.Sequential(
            utils.wn_xavier(nn.Linear(base_dim, 2*base_dim, bias=False)),
            Func(F.glu, dim=-1),
            utils.wn_xavier(nn.Linear(base_dim, 2*base_dim, bias=False)),
            Func(F.glu, dim=-1),
        )

    def forward(self, x):
        r = self.update(x)
        return x + r


class MemoryNet(nn.Module):
    def __init__(self, n_head=1, gamma=None):
        super().__init__()
        self.n_head = n_head
        if gamma is not None:
            self.gamma = torch.Parameter(torch.as_tensor(gamma))

    def forward(self, q, k, v, mask=None):
        # q: bxdxm
        # k: bxdxn
        # v: bxcxn
        # mask: bxmxn | 1xmxn
        # ret: bxcxm
        n_head = self.n_head
        if n_head > 1:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            b_size, d_size, m_size = q.size()
            c_size = v.size(1)
            q = q.view(b_size*n_head, d_size//n_head, m_size)
            k = k.view(b_size*n_head, d_size//n_head, -1)
            v = v.view(b_size*n_head, c_size//n_head, -1)

        d_size = q.size(1)
        qk = torch.matmul(q.transpose(-1, -2), k)/torch.sqrt(torch.as_tensor(d_size, dtype=q.dtype)) # bxmxn
        if mask is not None:
            sm = F.softmax(mask+qk, dim=-1) # bxmxn
        else:
            sm = F.softmax(qk, dim=-1) # bxmxn
        ret = (sm[:, None, :, :] * v[:, :, None, :]).sum(dim=-1) # bxcxmxn -> bxcxm
        if n_head > 1:
            ret = ret.view(b_size, c_size, m_size)
        if hasattr(self, 'gamma'):
            return self.gamma * ret
        else:
            return ret


if __name__ == "__main__":
    import numpy as np
    sa = MemoryNet(n_head=2)
    q = torch.randn(1,10,3)
    k = torch.randn(1,10,4)
    v = torch.randn(1,20,4)
    mask = torch.zeros(1, 3, 4)
    mask[:,:,-1] = -np.inf
    ret = sa(q,k,v, mask)
    print(ret.size())
    print(v)
    print(ret)