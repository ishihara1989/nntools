#include <cmath>
#include <limits>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void dpmatch_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> D,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R
){
    const auto B = blockIdx.x;
    const auto I = threadIdx.x;
    const auto x_size = D.size(1);
    const auto y_size = D.size(2);
    const auto T = x_size + y_size;
    for(int ij=0; ij<T; ij++){
        for(int i=I; i<T; i+=blockDim.x){
            const int j = ij-i;
            if(i==0){
                if(j==0){
                    R[B][i][j] = D[B][i][j];
                }
                else if(j>0 && j<y_size){
                    R[B][i][j] = R[B][i][j-1] + D[B][i][j];
                }
            }
            else if(i < x_size){
                if(j==0){
                    R[B][i][j] = R[B][i-1][j] + D[B][i][j];
                }
                else if(j>0 && j<y_size){
                    scalar_t rx = -R[B][i-1][j];
                    scalar_t ry = -R[B][i][j-1];
                    scalar_t rxy = -R[B][i-1][j-1];
                    scalar_t rmax = max(rx, max(ry, rxy));
                    scalar_t rsum = exp(rx-rmax)+exp(ry-rmax)+exp(rxy-rmax);
                    scalar_t softmin = log(rsum) + rmax;
                    R[B][i][j] = D[B][i][j] + softmin;
                }
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void dpmatch_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> D,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> E
){
    auto B = blockIdx.x;
    auto I = threadIdx.x;
    auto x_size = D.size(1);
    auto y_size = D.size(2);
    auto T = x_size + y_size;
    for(int ij=0; ij<T; ij++){
        for(int i=I; i<T; i+=blockDim.x){
            int j = ij-i;
            if(i==0){
                if(j==0){
                    E[B][x_size-1-i][y_size-1-j] = 1.0;
                }
                else if(j>0 && j<y_size){
                    scalar_t y0 = (R[B][x_size-1-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - D[B][x_size-1-i][y_size-j]);
                    scalar_t gy = exp(y0);
                    E[B][x_size-1-i][y_size-1-j] = gy * E[B][x_size-1-i][y_size-j];
                }
            }
            else if(i < x_size){
                if(j==0){
                    scalar_t x0 = (R[B][x_size-i][y_size-1-j] - R[B][x_size-1-i][y_size-1-j] - D[B][x_size-i][y_size-1-j]);
                    scalar_t gx = exp(x0);
                    E[B][x_size-1-i][y_size-1-j] = gx * E[B][x_size-i][y_size-1-j];
                }
                else if(j>0 && j<y_size){
                    scalar_t x0 = (R[B][x_size-i][y_size-1-j] - R[B][x_size-1-i][y_size-1-j] - D[B][x_size-i][y_size-1-j]);
                    scalar_t y0 = (R[B][x_size-1-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - D[B][x_size-1-i][y_size-j]);
                    scalar_t xy0 = (R[B][x_size-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - D[B][x_size-i][y_size-j]);
                    scalar_t gx = exp(x0);
                    scalar_t gy = exp(y0);
                    scalar_t gxy = exp(xy0);
                    E[B][x_size-1-i][y_size-1-j] = gx * E[B][x_size-i][y_size-1-j] + gy * E[B][x_size-1-i][y_size-j] + gxy * E[B][x_size-i][y_size-j];
                }
            }
        }
        __syncthreads();
    }
}


torch::Tensor dpmatch_cuda_forward(torch::Tensor D){
    int B = D.size(0);
    int N = D.size(1);
    int M = D.size(2);
    auto Rf = torch::zeros({B, N, M},
        torch::dtype(D.dtype())
        .layout(torch::kStrided)
        .device(D.device())
        .requires_grad(false));
    auto Rb = torch::zeros({B, N, M},
        torch::dtype(D.dtype())
        .layout(torch::kStrided)
        .device(D.device())
        .requires_grad(false));

    auto blocks = B;
    auto threads = 512;
    AT_DISPATCH_FLOATING_TYPES(D.type(), "dpmatch_forward_cuda", ([&] {
        dpmatch_cuda_forward_kernel<scalar_t><<<blocks,threads>>>(
            D.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return R;
}

torch::Tensor dpmatch_cuda_backward(torch::Tensor D, torch::Tensor R){
    int B = D.size(0);
    int N = D.size(1);
    int M = D.size(2);
    auto E = torch::zeros({B, N, M},
        torch::dtype(D.dtype())
        .layout(torch::kStrided)
        .device(D.device())
        .requires_grad(false));

    auto blocks = B;
    auto threads = 512;
    AT_DISPATCH_FLOATING_TYPES(D.type(), "dpmatch_backward_cuda", ([&] {
        dpmatch_cuda_backward_kernel<scalar_t><<<blocks,threads>>>(
            D.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            E.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return E;
}