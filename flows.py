import numpy as np
import scipy
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

__all__ = ['ChannelShuffle', 'AffineCoupling']

class ChannelShuffle(nn.Module):
    """
    a.k.a invertible 1x1 conv
    """
    def __init__(self, num_channels, rank=2, LU_decomposed=False):
        super().__init__()
        if rank<0 or rank > 3:
            raise ValueError("supported rank: 0 <= rank <= 3")
        self.rank = rank
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, x, invert):
        w_shape = self.w_shape
        if not self.LU:
            sizes=x.size()
            pixels = sizes[0] # prod(sizes[0], sizes[2:])
            if len(sizes) > 2:
                for s in sizes[2:]:
                    pixels *= s
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            na = [1 for _ in range(len(sizes)-2)]
            if not invert:
                weight = self.weight.view(*w_shape, *na)
            else:
                weight = torch.inverse(self.weight.double()).float().view(*w_shape, *na)
            return weight, dlogdet
        else:
            self.p = self.p.to(x.device)
            self.sign_s = self.sign_s.to(x.device)
            self.l_mask = self.l_mask.to(x.device)
            self.eye = self.eye.to(x.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            sizes = x.size()
            pixels = sizes[0]
            if len(sizes) > 2:
                for s in sizes[2:]:
                    pixels *= s
            dlogdet = torch.sum(self.log_s) * pixels
            na = [1 for _ in range(len(sizes)-2)]
            if not invert:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else: # @@
                li = torch.inverse(l.double()).float()
                ui = torch.inverse(u.double()).float()
                w = torch.matmul(ui, torch.matmul(li, self.p.inverse()))
            return w.view(*w_shape, *na), dlogdet

    def forward(self, x, logdet=None, invert=False):
        weight, dlogdet = self.get_weight(x, invert)
        if self.rank == 0:
            func = F.linear
        elif self.rank == 1:
            func = F.conv1d
        elif self.rank == 2:
            func = F.conv2d
        elif self.rank == 3:
            func = F.conv3d
        else:
            raise ValueError(f"invalid rank: {self.rank}")

        if not invert:
            z = func(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
                return z, logdet
            else:
                return z
        else:
            z = func(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
                return z, logdet
            else:
                return z

def test_iconv(use_lu=True):
    c_size = 2
    b_size = 1
    s_size = 3
    iconv = ChannelShuffle(c_size, 0, use_lu)
    x = torch.randn(b_size,c_size)
    z = iconv(x)
    x2 = iconv(z, invert=True)
    print(f'0d: {(x-x2).pow(2).sum()}')
    iconv = ChannelShuffle(c_size, 1, use_lu)
    x = torch.randn(b_size,c_size,s_size)
    z = iconv(x)
    x2 = iconv(z, invert=True)
    print(f'1d: {(x-x2).pow(2).sum()}')
    iconv = ChannelShuffle(c_size, 2, use_lu)
    x = torch.randn(b_size,c_size,s_size,s_size)
    z = iconv(x)
    x2 = iconv(z, invert=True)
    print(f'2d: {(x-x2).pow(2).sum()}')
    iconv = ChannelShuffle(c_size, 3, use_lu)
    x = torch.randn(b_size,c_size,s_size,s_size,s_size)
    z = iconv(x)
    x2 = iconv(z, invert=True)
    print(f'3d: {(x-x2).pow(2).sum()}')


class AffineCoupling(nn.Module):
    def __init__(self, out_channels, inner_channels=None, layers=3, rank=2, zero_init=True, net=None):
        super().__init__()
        if rank<0 or rank > 3:
            raise ValueError("supported rank: 0 <= rank <= 3")
        if net is not None:
            self.net = net
        else:
            if inner_channels is None:
                raise ValueError("either `net` or `inner_channels` must be given")
            kwargs = {} if rank==0 else {'kernel_size': 3, 'padding': 1}
            if rank == 0:
                conv = nn.Linear
            elif rank == 1:
                conv = nn.Conv1d
            elif rank == 2:
                conv = nn.Conv2d
            elif rank == 3:
                conv = nn.Conv3d
            else:
                raise ValueError(f"invalid rank: {rank}")
            modules = [utils.wn_xavier(conv(out_channels, inner_channels, **kwargs)), nn.ELU()]
            for _ in range(layers):
                modules.append(utils.wn_xavier(conv(inner_channels, inner_channels, **kwargs)))
                modules.append(nn.ELU())
            end = conv(inner_channels, 2*out_channels, **kwargs)
            if zero_init:
                end.weight.data.zero_()
                end.bias.data.zero_()
            modules.append(end)
            self.net = nn.Sequential(*modules)
        self.alpha = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.rank = rank

    def forward(self, x, logdet=None, invert=False):
        x_0, x_1 = x.chunk(2, dim=1)
        r = self.net(x_0)
        log_s, t = r.chunk(2, dim=1)
        s_size = log_s.size(1)
        alpha = self.alpha.view(1, s_size, *[1 for _ in range(self.rank)])
        beta = self.beta.view(1, s_size, *[1 for _ in range(self.rank)])
        log_s = alpha*torch.tanh(log_s) + beta
        
        if not invert:
            s = torch.exp(log_s)
            z = torch.cat([x_0, s*x_1+t], dim=1)
            if logdet is not None:
                dlogdet = log_s.sum()
                logdet = logdet + dlogdet
                return z, logdet
            else:
                return z
        else:
            si = torch.exp(-log_s)
            z = torch.cat([x_0, (x_1-t)*si], dim=1)
            if logdet is not None:
                dlogdet = log_s.sum()
                logdet = logdet - dlogdet
                return z, logdet
            else:
                return z

def test_ac():
    b_size = 20
    c_size = 4
    h_size = 8
    s_size = 10
    ac = AffineCoupling(c_size, h_size, rank=0, zero_init=False)
    x = torch.randn(b_size, 2*c_size)
    z, ld = ac(x, 0)
    x2 = ac(z, invert=True)
    print(f'0d: {(x-x2).pow(2).sum()}, {ld/b_size}')
    ac = AffineCoupling(c_size, h_size, rank=1, zero_init=False)
    x = torch.randn(b_size,2*c_size,s_size)
    z, ld = ac(x, 0)
    x2 = ac(z, invert=True)
    print(f'1d: {(x-x2).pow(2).sum()}, {ld/b_size/s_size}')
    ac = AffineCoupling(c_size, h_size, rank=2, zero_init=False)
    x = torch.randn(b_size,2*c_size,s_size,s_size)
    z, ld = ac(x, 0)
    x2 = ac(z, invert=True)
    print(f'2d: {(x-x2).pow(2).sum()}, {ld/b_size/s_size**2}')
    ac = AffineCoupling(c_size, h_size, rank=3, zero_init=False)
    x = torch.randn(b_size,2*c_size,s_size,s_size,s_size)
    z, ld = ac(x, 0)
    x2 = ac(z, invert=True)
    print(f'3d: {(x-x2).pow(2).sum()}, {ld/b_size/s_size**2}')

    net = nn.Linear(c_size, 2*c_size)
    ac = AffineCoupling(c_size, rank=0, net=net)
    x = torch.randn(b_size, 2*c_size)
    z, ld = ac(x, 0)
    x2 = ac(z, invert=True)
    print(f'net: {(x-x2).pow(2).sum()}, {ld/b_size}')


class ARUnit(nn.Module):
    def __init__(self):
        super(ARUnit, self).__init__()

    def forward(self, x):
        return
    
    def ar(self, x):
        return

    def reset(self):
        pass

class Macow(nn.Module):
    def __init__(self, output_dim, dim, inner_dim, kernel=2, layers=2, reversed=False):
        super(Macow, self).__init__()


if __name__ == "__main__":
    import sys
    if len(sys.argv)>1:
        kind = sys.argv[1]
    else:
        kind = 'all'

    is_all = kind == 'all'
    if kind == 'iconv' or is_all:
        print('iconv', '*'*50)
        test_iconv(True)
        test_iconv(False)
        print('*'*60)

    if kind == 'affine' or is_all:
        print('affile', '*'*50)
        test_ac()
        print('*'*60)