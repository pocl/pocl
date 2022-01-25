extern "C"
__global__ void pocl_add32(unsigned *x, unsigned *y, unsigned *out)
{
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      out[tid] = x[tid] + y[tid];
}

extern "C"
__global__ void pocl_mul32(unsigned *x, unsigned *y, unsigned *out)
{
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      out[tid] = x[tid] * y[tid];
}
