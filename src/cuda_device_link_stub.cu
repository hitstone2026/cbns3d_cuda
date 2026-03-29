// Triggers CMake's CUDA device link for executables that link block3d_cuda (RDC).
// Not part of block3d_cuda — excluded from the static library's .cu glob.
__global__ void cbns3d_cuda_device_link_stub_kernel() {}
