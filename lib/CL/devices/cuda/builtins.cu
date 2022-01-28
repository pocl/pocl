
// *************************************************************
// simple builtin kernels

extern "C"
__global__ void pocl_add_i32(const int* __restrict x, const int* __restrict y, int* __restrict out)
{
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      out[tid] = x[tid] + y[tid];
}


extern "C"
__global__ void pocl_mul_i32(const int* __restrict x, const int* __restrict y, int* __restrict out)
{
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      out[tid] = x[tid] * y[tid];
}

// *************************************************************
// DNN builtin kernel dummy example

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

// *************************************************************
// builtin kernel example using local memory

#define TILE_DIM 16

// M, N, K must be divisible by TILE_DIM
// M x N = global size
extern "C"
__global__ void pocl_sgemm_local_f32(
   const float* __restrict A,
   const float* __restrict B,
   float* __restrict C,
   unsigned M, unsigned N, unsigned K
)
{
    float CValue = 0;
    size_t ARows = M;
    size_t ACols = K;
    size_t BRows = K;
    size_t BCols = N;
    size_t CRows = M;
    size_t CCols = N;

    size_t Row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t Col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (size_t k = 0; k < K/TILE_DIM; k++) {

         As[threadIdx.y][threadIdx.x] = A[Row*K + k*TILE_DIM + threadIdx.x];
         Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];

         __syncthreads();

         for (size_t n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    C[(Row*CCols) + Col] = CValue;
}


// *************************************************************

// this code requires Tensor Cores
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

#include <mma.h>
using namespace nvcuda;

// NOTE matrix size must be multiples of 16 for wmma code to work

// The only dimension multiples currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
extern "C"
__global__ void pocl_sgemm_scale_tensor_f16f16f32(half* __restrict a, half* __restrict b, float* __restrict c,
                                        unsigned M, unsigned N, unsigned K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

// same kernel, but without scaling
extern "C"
__global__ void pocl_sgemm_tensor_f16f16f32(half* __restrict a, half* __restrict b, float* __restrict c,
                                  unsigned M, unsigned N, unsigned K) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      int cRow = warpM * WMMA_M;
      int cCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);

      }
   }
}

#endif
