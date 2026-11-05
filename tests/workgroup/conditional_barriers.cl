__kernel void
test_kernel (void)
{
  unsigned group_id = get_group_id (0);
  unsigned local_id = get_local_id (0);

  printf ("LOCAL_ID=%d before if\n", local_id);
  /* uniform, but unknown at compile time, unlike the local size of a
     specialized work-group function */
  if (get_num_groups(0) < 100)
  {
      printf ("LOCAL_ID=%d inside if\n", local_id);
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  printf ("LOCAL_ID=%d after if\n", local_id);
}
