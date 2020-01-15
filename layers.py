from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

__all__ = ['Func', 'DenseResBlocks1d', 'InfusedResBlock1d', 'StackedInfusedResBlock1d', 'LinearResBlock', 'MemoryNet', 'VTLN']

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
        super().__init__(channels)
        self.update = nn.Sequential(
            nn.ReflectionPad2d(1),
            utils.wn_xavier(nn.Conv2d(channels, 2*channels, 3, bias=False)),
            Func(F.glu, dim=1),
            nn.ReflectionPad2d(1),
            utils.wn_xavier(nn.Conv2d(channels, 2*channels, 3, bias=False)),
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
    def forward(self, x, scale=None, bias=None):
        rs = x
        if scale is not None:
            scales = scale.chunk(self.n_layers, dim=1)
        if bias is not None:
            biases = bias.chunk(self.n_layers, dim=1)
        for i in range(self.n_layers):
            rs = self.convs[i](rs)
            if scale is not None:
                rs = rs * scales[i]
            if bias is not None:
                rs = rs + biases[i]
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
            self.gamma = nn.Parameter(torch.as_tensor(float(gamma)))

    def forward(self, q, k, v, mask=None, alpha=1.0):
        # q: bxdxm
        # k: bxdxn
        # v: bxcxn
        # mask: bxmxn | 1xmxn
        # ret: bxcxm
        n_head = self.n_head
        q_shape = q.size()
        k_shape = k.size()
        v_shape = v.size()
        if n_head > 1:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            q = q.view(*q_shape[:-2], n_head, q_shape[-2]//n_head, q_shape[-1])
            k = k.view(*k_shape[:-2], n_head, k_shape[-2]//n_head, k_shape[-1])
            v = v.view(*v_shape[:-2], n_head, v_shape[-2]//n_head, v_shape[-1])
            if mask is not None:
                mask_shape = mask.size()
                mask = mask.unsqueeze(-3)
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)

        qk = alpha * torch.matmul(q.transpose(-1, -2), k) # bxhxmxn
        if mask is not None:
            sm = F.softmax(mask+qk, dim=-1) # bxmxn
        else:
            sm = F.softmax(qk, dim=-1) # bxmxn
        ret = (sm.unsqueeze(-3) * v.unsqueeze(-2)).sum(dim=-1) # bxcxmxn -> bxcxm
        if n_head > 1:
            ret = ret.view(-1, v_shape[-2], q_shape[-1])
        if hasattr(self, 'gamma'):
            return self.gamma * ret
        else:
            return ret

    def softmax(self, q, k, alpha=1.0, mask=None):
        n_head = self.n_head
        q_shape = q.size()
        k_shape = k.size()
        if n_head > 1:
            q, k = q.contiguous(), k.contiguous()
            q = q.view(*q_shape[:-2], n_head, q_shape[-2]//n_head, q_shape[-1])
            k = k.view(*k_shape[:-2], n_head, k_shape[-2]//n_head, k_shape[-1])
            if mask is not None:
                mask_shape = mask.size()
                mask = mask.unsqueeze(-3)
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        qk = alpha * torch.matmul(q.transpose(-1, -2), k) # bxhxmxn
        if mask is not None:
            return F.softmax(mask+qk, dim=-1) # bxmxn
        else:
            return F.softmax(qk, dim=-1) # bxmxn


class VTLN(nn.Module):
    def __init__(self, n_mcep):
        super().__init__()
        a = np.zeros((n_mcep, n_mcep, 2*n_mcep+1), dtype=np.float32)
        for k in range(1, n_mcep+1):
            for l in range(1, n_mcep+1):
                for n in range(n_mcep+1):
                    if l-k<=n<=l:
                        a[k-1,l-1,2*n+k-l] = comb(l,n) * comb(k+n-1, l-1) * (-1)**(n+k+l)
        self.register_buffer('a_3d', torch.Tensor(a))
        self.n_mcep = n_mcep

    def forward(self, mcep, alpha):
        # mcep: bxmxt
        # alpha: bxt
        b_size = alpha.size(0)
        t_size = alpha.size(-1)
        one = torch.ones(b_size, t_size, dtype=alpha.dtype, device=alpha.device)
        ones = one[..., None].expand([*one.size(), 2*self.n_mcep]) * alpha[..., None] #bxtxd
        alpha_v = torch.cumprod(torch.cat([one[..., None], ones], dim=-1), dim=-1)
        warp_mat = torch.einsum('btd,xyd->btxy', alpha_v, self.a_3d) # bxtxmxm
        return torch.einsum('bmt,btnm->bnt', mcep, warp_mat)


if __name__ == "__main__":
    import numpy as np
    import sys
    if len(sys.argv)>1:
        kind = sys.argv[1]
    else:
        kind = 'all'

    is_all = kind == 'all'
    if kind == 'memory' or is_all:
        sa = MemoryNet(n_head=2)
        q = torch.randn(7,1,10,3)
        k = torch.randn(1,11,10,4)
        v = torch.randn(1,11,20,4)
        mask = torch.zeros(7, 11, 3, 4)
        mask[...,-1] = -np.inf
        ret = sa(q,k,v, mask)
        print(ret.size())
        print(sa.softmax(q, k).size())
    if kind == 'vtln' or is_all:
        b_size = 4
        m_size = 40
        t_size = 64
        vtln = VTLN(m_size)
        z = torch.randn(b_size, m_size, t_size)
        alpha = 0.2 * torch.ones(b_size, t_size)
        ret = vtln(z, alpha)
        print(ret.size())