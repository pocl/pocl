__kernel void
test_kernel (__global int *data)
{
  // Not get_local_size (0) > 1: that folds away before the barrier passes.
  if (data[0] > 0)
    barrier (CLK_LOCAL_MEM_FENCE);
  printf ("LOCAL_ID=%d,%d,%d\n", (int)get_local_id (0), (int)get_local_id (1),
          (int)get_local_id (2));
}
