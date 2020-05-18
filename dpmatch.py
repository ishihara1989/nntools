from pathlib import Path

import numpy as np
import torch
from numba import jit
from torch.autograd import Function

from torch.utils.cpp_extension import load


class Loader:
    def __init__(self):
        self.DTW_CUDA_SUCCEED = False
        self.dpmatch_cuda = None

    def load(self):
        if self.dpmatch_cuda is None:
            try:
                root = Path(__file__).resolve().parent
                self.dpmatch_cuda = load(
                    name="dpmatch_cuda",
                    sources=[
                        root / "dpmatch_src/dpmatch_cuda.cpp",
                        root / "dpmatch_src/dpmatch_cuda_kernel.cu",
                    ],
                )
                self.DTW_CUDA_SUCCEED = True
            except Exception as e:
                self.DTW_CUDA_SUCCEED = False
                print(e)
                print("cuda DTW load failed")


loader = Loader()


@jit(nopython=True)
def _dp_forward(D):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.copy(D)
    for k in range(B):
        for i in range(1, N):
            R[k, i, 0] += R[k, i - 1, 0]
        for j in range(1, M):
            R[k, 0, j] += R[k, 0, j - 1]
            for i in range(1, N):
                r0 = R[k, i - 1, j - 1]
                r1 = R[k, i - 1, j]
                r2 = R[k, i, j - 1]
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmax = np.log(rsum) + rmax
                R[k, i, j] += softmax
    return R


@jit(nopython=True)
def _dp_backward(grad_R, D, R):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    E = np.copy(grad_R)
    for k in range(B):
        for i in range(N - 2, -1, -1):
            a0 = R[k, i + 1, M - 1] - R[k, i, M - 1] - D[k, i + 1, M - 1]
            a = np.exp(a0)
            E[k, i, M - 1] += E[k, i + 1, M - 1] * a
        for j in range(M - 2, -1, -1):
            b0 = R[k, N - 1, j + 1] - R[k, N - 1, j] - D[k, N - 1, j + 1]
            b = np.exp(b0)
            E[k, N - 1, j] += E[k, N - 1, j + 1] * b
            for i in range(N - 2, -1, -1):
                a0 = R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]
                b0 = R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]
                c0 = R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] += (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )
    return E


class _SoftDP(Function):
    @staticmethod
    def forward(ctx, D):
        dev = D.device
        dtype = D.dtype
        D_ = D.detach().cpu().numpy()
        R = torch.Tensor(_dp_forward(D_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R)
        return R

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g = grad_output.detach().cpu().numpy()
        E = torch.Tensor(_dp_backward(g, D_, R_)).to(dev).type(dtype)
        return E


class SoftDPMatch(torch.nn.Module):
    def __init__(self, gamma=1.0, t=1.0):
        super().__init__()
        self.gamma = gamma
        self.t = t

    def forward(self, d):
        d = d / self.gamma
        _, N, M = d.size()
        f = _SoftDP.apply(d)
        b = _SoftDP.apply(
            d[:, torch.arange(N - 1, -1, -1), :][:, :, torch.arange(M - 1, -1, -1)]
        )
        logit = (
            f
            + b[:, torch.arange(N - 1, -1, -1), :][:, :, torch.arange(M - 1, -1, -1)]
            - d
        )
        return self.gamma * (logit - logit.max()) / self.t


if __name__ == "__main__":
    from utils import pdist

    gamma = torch.nn.Parameter(torch.as_tensor(10.0))
    t = torch.nn.Parameter(torch.as_tensor(10.0))
    fun = SoftDPMatch(gamma, t)
    x = torch.arange(3, dtype=torch.float32)[None, None, :]
    y = torch.arange(4, dtype=torch.float32)[None, None, :]
    y = torch.nn.Parameter(y)
    opt = torch.optim.Adam([gamma, t, y], 1e-3)
    gt = torch.as_tensor(
        [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
    ).unsqueeze(0)
    d = pdist(x, y)

    for i in range(10000):
        opt.zero_grad()
        d = pdist(x, y)
        logit = fun(-d)
        loss = (d * logit.exp()).mean()
        loss.backward()
        opt.step()
        print(f"{loss.data=}")
    print(f"{y=}")
    print(f"{gamma=}")
    print(f"{t=}")
    print(f"{logit.exp()=}")
