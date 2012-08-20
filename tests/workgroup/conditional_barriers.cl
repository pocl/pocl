__kernel void
test_kernel (void)
{
  unsigned group_id = get_group_id (0);
  unsigned local_id = get_local_id (0);

  printf ("LOCAL_ID=%d before if\n", local_id);
  if (get_local_size(0) < 100)
  {
      printf ("LOCAL_ID=%d inside if\n", local_id);
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  printf ("LOCAL_ID=%d after if\n", local_id);
}
