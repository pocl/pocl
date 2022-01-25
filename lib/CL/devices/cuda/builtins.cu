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

extern "C"
__global__ void pocl_dnn_conv2d_int8_relu(
  char* __restrict placeholder,
  char* __restrict placeholder1,
  char* __restrict compute,
  int* __restrict placeholder2,
  int* __restrict placeholder3,
  int* __restrict placeholder4,
  int* __restrict placeholder5,
  int* __restrict placeholder6
)
{
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      placeholder6[tid] = placeholder2[tid] * placeholder3[tid];
      placeholder5[tid] = placeholder4[tid] * placeholder2[tid];
}
