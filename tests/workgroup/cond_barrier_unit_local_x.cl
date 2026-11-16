__kernel void
test_kernel (void)
{
  if (get_local_size (0) > 1)
    barrier (CLK_LOCAL_MEM_FENCE);
  printf ("LOCAL_ID=%d,%d\n", (int)get_local_id (0), (int)get_local_id (1));
}
