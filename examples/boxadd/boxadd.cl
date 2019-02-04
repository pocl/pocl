kernel void
boxadd (__global const float *A,
	__global const float *B,
	__global float *C,
	int SX, int SY, int SZ)
{
  size_t X = get_global_id(0);
  if (X >= SX) return;
  size_t Y = get_global_id(1);
  if (Y >= SY) return;
  size_t Z = get_global_id(2);
  if (Z >= SZ) return;

  size_t Idx = Z*SY*SX + Y*SX + X;

  C[Idx] = A[Idx] + B[Idx];
}
