kernel void
matadd (__global const float *A,
	__global const float *B,
	__global float *C)
{
  size_t X = get_global_id(0);
  size_t Y = get_global_id(1);
  size_t Idx = Y*get_global_size(0) + X;

  C[Idx] = A[Idx] + B[Idx];
}
