#include <torch/extension.h>
#include <vector>

torch::Tensor dpmatch_cuda_forward(torch::Tensor D);
torch::Tensor dpmatch_cuda_backward(torch::Tensor D, torch::Tensor R);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dpmatch_forward(torch::Tensor D){
    CHECK_INPUT(D);
    TORCH_CHECK(D.sizes().size()==3, "D must have 3 dims");
    return dpmatch_cuda_forward(D);
}

torch::Tensor dpmatch_backward(torch::Tensor D, torch::Tensor R){
    CHECK_INPUT(R);
    CHECK_INPUT(D);
    TORCH_CHECK(D.sizes().size()==3, "D must have 3 dims");
    TORCH_CHECK(R.sizes().size()==3, "R must have 3 dims");
    return dpmatch_cuda_backward(D, R);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dpmatch_forward, "Soft DTW forward (CUDA)");
  m.def("backward", &dpmatch_backward, "Soft DTW backward (CUDA)");
}