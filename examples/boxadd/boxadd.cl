kernel void
boxadd (__global const float *A,
	__global const float *B,
	__global float *C)
{
  size_t X = get_global_id(0);
  size_t Y = get_global_id(1);
  size_t Z = get_global_id(2);
  size_t Idx =
    Z*get_global_size(1)*get_global_size(0) + Y*get_global_size(0) + X;

  C[Idx] = A[Idx] + B[Idx];
}
