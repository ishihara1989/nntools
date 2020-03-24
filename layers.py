import numpy as np
from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

__all__ = [
    "Func",
    "DenseResBlocks1d",
    "InfusedResBlock1d",
    "StackedInfusedResBlock1d",
    "LinearResBlock",
    "MemoryNet",
    "VTLN",
]


class Func(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, **self.kwargs)

    def __repr__(self):
        return "Func(func={})".format(self.func.__name__)


class CausalResBlock1d(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        if hidden_channels is None:
            hidden_channels = in_channels
        self.pad = nn.ConstantPad1d((2, 0), 0.0)
        self.input = nn.Sequential(
            utils.wn_xavier(nn.Conv1d(in_channels, 2 * hidden_channels, 3)),
            Func(F.glu, dim=1),
        )
        self.output = nn.Sequential(
            utils.wn_xavier(nn.Conv1d(hidden_channels, 2 * in_channels, 1)),
            Func(F.glu, dim=1),
        )

    def forward(self, x, scale=None, bias=None):
        rs = self.input(self.pad(x))
        if scale is not None:
            rs = rs * scale
        if bias is not None:
            rs = rs + bias
        rs = self.output(rs)
        return x + rs

    def step(self, x, scale=None, bias=None):
        rs = self.input(x)
        if scale is not None:
            rs = rs * scale
        if bias is not None:
            rs = rs + bias
        rs = self.output(rs)
        return x[..., -1:] + rs


class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__(channels)
        self.update = nn.Sequential(
            nn.ReflectionPad2d(1),
            utils.wn_xavier(nn.Conv2d(channels, 2 * channels, 3, bias=False)),
            Func(F.glu, dim=1),
            nn.ReflectionPad2d(1),
            utils.wn_xavier(nn.Conv2d(channels, 2 * channels, 3, bias=False)),
            Func(F.glu, dim=1),
        )

    def forward(self, x):
        r = self.update(x)
        return x + r


class DenseResBlocks1d(nn.Module):
    def __init__(self, n_channels, dilations, use_skip=True):
        super(DenseResBlocks1d, self).__init__()
        self.use_skip = use_skip
        self.n_layers = len(dilations)
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            self.convs.append(
                nn.Sequential(
                    nn.ReplicationPad1d(dilation),
                    utils.wn_xavier(
                        nn.Conv1d(n_channels, 2 * n_channels, 3, dilation=dilation)
                    ),
                    Func(F.glu, dim=1),
                )
            )
            last_c = (
                4 * n_channels if use_skip and i < self.n_layers - 1 else 2 * n_channels
            )
            self.skips.append(
                nn.Sequential(
                    utils.wn_xavier(nn.Conv1d(n_channels, last_c, 1)),
                    Func(F.glu, dim=1),
                )
            )

    def forward(self, x, scale=None, bias=None):
        rs = x
        if scale is not None:
            scales = scale.chunk(self.n_layers, dim=1)
        if bias is not None:
            biases = bias.chunk(self.n_layers, dim=1)

        if self.use_skip:
            for i in range(self.n_layers):
                rs = self.convs[i](rs)
                if scale is not None:
                    rs = rs * scales[i]
                if bias is not None:
                    rs = rs + biases[i]
                rs = self.skips[i](rs)

                if i < self.n_layers - 1:
                    rs, skip = rs.chunk(2, dim=1)
                    if i == 0:
                        out = skip
                    else:
                        out = out + skip
                else:
                    out = out + rs
            return out
        else:
            for i in range(self.n_layers):
                cond = self.convs[i](rs)
                if scale is not None:
                    cond = cond * scales[i]
                if bias is not None:
                    cond = cond + biases[i]
                rs = rs + self.skips[i](cond)
            return rs


class InfusedResBlock1d(nn.Module):
    def __init__(self, base_dim):
        super(InfusedResBlock1d, self).__init__()
        self.update = nn.Sequential(
            utils.wn_xavier(
                nn.Conv1d(base_dim, 2 * base_dim, 3, padding=1, bias=False)
            ),
            Func(F.glu, dim=1),
            utils.wn_xavier(
                nn.Conv1d(base_dim, 2 * base_dim, 3, padding=1, bias=False)
            ),
            Func(F.glu, dim=1),
        )
        self.alpha = nn.Parameter(torch.ones(1, base_dim, 1))

    def forward(self, x, w, b):
        r = self.update(self.alpha * torch.sigmoid(w) * x + b)
        return x + r


class StackedInfusedResBlock1d(nn.Module):
    def __init__(self, base_dim, layers):
        super(StackedInfusedResBlock1d, self).__init__()
        self.layers = nn.ModuleList(
            [InfusedResBlock1d(base_dim) for _ in range(layers)]
        )

    def forward(self, x, cond):
        l = len(self.layers)
        w, b = torch.chunk(cond, 2, dim=1)
        ws = torch.chunk(w, l, dim=1)
        bs = torch.chunk(b, l, dim=1)
        r = x
        for i, layer in enumerate(self.layers):
            r = layer(r, ws[i][:, :, None], bs[i][:, :, None])
        return r


class LinearResBlock(nn.Module):
    def __init__(self, base_dim):
        super(LinearResBlock, self).__init__()
        self.update = nn.Sequential(
            utils.wn_xavier(nn.Linear(base_dim, 2 * base_dim, bias=False)),
            Func(F.glu, dim=-1),
            utils.wn_xavier(nn.Linear(base_dim, 2 * base_dim, bias=False)),
            Func(F.glu, dim=-1),
        )

    def forward(self, x):
        r = self.update(x)
        return x + r


class ComplexComv1d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1
    ):
        super().__init__()
        n = np.sqrt(2 / (in_channel + out_channel) / 3)
        wr = torch.randn(out_channel, in_channel, kernel_size) * n
        wi = torch.randn(out_channel, in_channel, kernel_size) * n
        g = (wr ** 2 + wi ** 2).sum(dim=[1, 2], keepdim=True).sqrt()
        wr /= g
        wi /= g
        self.weight_g = torch.nn.Parameter(g)
        self.weight_r = torch.nn.Parameter(wr)
        self.weight_i = torch.nn.Parameter(wi)
        self.bias = torch.zeros(out_channel * 2)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        wr = self.weight_g * self.weight_r
        wi = self.weight_g * self.weight_i
        w = torch.cat([torch.cat([wr, -wi], dim=1), torch.cat([wi, wr], dim=1)], dim=0)
        return F.conv1d(
            x,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class STFT(nn.Module):
    def __init__(self, win_length, hop_length=None, n_fft=None):
        super().__init__()
        self.window = torch.hamming_window(win_length)
        if hop_length is None:
            hop_length = win_length // 4
        if n_fft is None:
            n_fft = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x):
        z = torch.stft(
            x, self.n_fft, hop_length=self.hop_length, window=self.window, center=True
        )
        z = z.permute(0, 3, 1, 2).contiguous()
        b, _, c, t = z.size()
        return z.view(b, 2 * c, t)


class ISTFT(nn.Module):
    def __init__(self, win_length, hop_length=None, n_fft=None):
        super().__init__()
        window = torch.hamming_window(win_length)
        if hop_length is None:
            hop_length = win_length // 4
        if n_fft is None:
            n_fft = win_length
        self.win_length = win_length
        self.window = window
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.ola = torch.eye(win_length)[:, None, :]

    def forward(self, z):
        b, c, t = z.size()
        z = z.contiguous().view(b, 2, c // 2, t).permute(0, 3, 2, 1).contiguous()
        sig = torch.irfft(z, signal_ndim=1, signal_sizes=(self.n_fft,))
        sig = sig[:, :, : self.win_length] * self.window
        sig = sig.permute(0, 2, 1)
        sig = F.conv_transpose1d(sig, self.ola, stride=self.hop_length, padding=0)[
            :, 0, self.win_length // 2 : -self.win_length // 2
        ]
        win = self.window.pow(2).view(self.n_fft, 1).repeat((1, z.size(1))).unsqueeze(0)
        win = F.conv_transpose1d(win, self.ola, stride=self.hop_length, padding=0)[
            :, 0, self.win_length // 2 : -self.win_length // 2
        ]
        return sig / win


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
            q = q.view(*q_shape[:-2], n_head, q_shape[-2] // n_head, q_shape[-1])
            k = k.view(*k_shape[:-2], n_head, k_shape[-2] // n_head, k_shape[-1])
            v = v.view(*v_shape[:-2], n_head, v_shape[-2] // n_head, v_shape[-1])
            if mask is not None:
                mask = mask.unsqueeze(-3)
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)

        qk = alpha * torch.matmul(q.transpose(-1, -2), k)  # bxhxmxn
        if mask is not None:
            sm = F.softmax(mask + qk, dim=-1)  # bxmxn
        else:
            sm = F.softmax(qk, dim=-1)  # bxmxn
        ret = (sm.unsqueeze(-3) * v.unsqueeze(-2)).sum(dim=-1)  # bxcxmxn -> bxcxm
        if n_head > 1:
            ret = ret.view(-1, v_shape[-2], q_shape[-1])
        if hasattr(self, "gamma"):
            return self.gamma * ret
        else:
            return ret

    def softmax(self, q, k, alpha=1.0, mask=None):
        n_head = self.n_head
        q_shape = q.size()
        k_shape = k.size()
        if n_head > 1:
            q, k = q.contiguous(), k.contiguous()
            q = q.view(*q_shape[:-2], n_head, q_shape[-2] // n_head, q_shape[-1])
            k = k.view(*k_shape[:-2], n_head, k_shape[-2] // n_head, k_shape[-1])
            if mask is not None:
                mask = mask.unsqueeze(-3)
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        qk = alpha * torch.matmul(q.transpose(-1, -2), k)  # bxhxmxn
        if mask is not None:
            return F.softmax(mask + qk, dim=-1)  # bxmxn
        else:
            return F.softmax(qk, dim=-1)  # bxmxn


class VTLN(nn.Module):
    def __init__(self, n_mcep):
        super().__init__()
        a = np.zeros((n_mcep, n_mcep, 2 * n_mcep + 1), dtype=np.float32)
        for k in range(1, n_mcep + 1):
            for l in range(1, n_mcep + 1):
                for n in range(n_mcep + 1):
                    if l - k <= n <= l:
                        a[k - 1, l - 1, 2 * n + k - l] = (
                            comb(l, n) * comb(k + n - 1, l - 1) * (-1) ** (n + k + l)
                        )
        self.register_buffer("a_3d", torch.Tensor(a))
        self.n_mcep = n_mcep

    def forward(self, mcep, alpha):
        # mcep: bxmxt
        # alpha: bxt
        b_size = alpha.size(0)
        t_size = alpha.size(-1)
        one = torch.ones(b_size, t_size, dtype=alpha.dtype, device=alpha.device)
        ones = (
            one[..., None].expand([*one.size(), 2 * self.n_mcep]) * alpha[..., None]
        )  # bxtxd
        alpha_v = torch.cumprod(torch.cat([one[..., None], ones], dim=-1), dim=-1)
        warp_mat = torch.einsum("btd,xyd->btxy", alpha_v, self.a_3d)  # bxtxmxm
        return torch.einsum("bmt,btnm->bnt", mcep, warp_mat)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        kind = sys.argv[1]
    else:
        kind = "all"

    is_all = kind == "all"
    if kind == "memory" or is_all:
        sa = MemoryNet(n_head=2)
        q = torch.randn(7, 1, 10, 3)
        k = torch.randn(1, 11, 10, 4)
        v = torch.randn(1, 11, 20, 4)
        mask = torch.zeros(7, 11, 3, 4)
        mask[..., -1] = -np.inf
        ret = sa(q, k, v, mask)
        print(ret.size())
        print(sa.softmax(q, k).size())

    if kind == "vtln" or is_all:
        b_size = 4
        m_size = 40
        t_size = 64
        vtln = VTLN(m_size)
        z = torch.randn(b_size, m_size, t_size)
        alpha = 0.2 * torch.ones(b_size, t_size)
        ret = vtln(z, alpha)
        print(ret.size())

    if kind == "stft" or is_all:
        import torch.optim

        x = torch.randn(1, 32, requires_grad=True)

        class Model(nn.Module):
            def __init__(self, n=8, h=32):
                super().__init__()
                self.stft = STFT(n, n // 4)
                self.istft = ISTFT(n, n // 4)
                self.layers = nn.Sequential(
                    ComplexComv1d(n // 2 + 1, h, 3, padding=1),
                    nn.GELU(),
                    ComplexComv1d(h, h, 3, padding=1),
                    nn.GELU(),
                    ComplexComv1d(h, n // 2 + 1, 3, padding=1),
                )

            def forward(self, x):
                z = self.stft(x)
                z = self.layers(z)
                return z

        stft = STFT(8, 2)
        istft = ISTFT(8, 2)
        z = stft(x)
        print(z.size())
        c = ComplexComv1d(5, 5, 3, padding=1)

        z2 = c(z)
        y = istft(z)
        y2 = istft(z2)
        print(x.size(), y.size())
        print((x - y).pow(2).sum())
        print(y2)

        x = nn.Parameter(torch.randn(1, 512, requires_grad=True))
        model = Model(16, 32)
        opt = torch.optim.Adam(list(model.parameters()) + [x], lr=1e-4)

        tgt = torch.zeros(1, 9, 1) + 1e-8
        tgt[0, 2, 0] = 1
        print(tgt.data)
        for k, v in model.state_dict().items():
            print(k, v.size())

        for n in range(50000):
            opt.zero_grad()
            y = model(x)
            y = model.istft(y)
            y = model.stft(y)
            r, i = y.chunk(2, dim=1)
            a = (r ** 2 + i ** 2).clamp(1e-8)
            loss = (a.log() - tgt.log()).pow(2).mean()
            loss.backward()
            if n % 100 == 99:
                print(loss.data)
            opt.step()

        print(a.data.T)
