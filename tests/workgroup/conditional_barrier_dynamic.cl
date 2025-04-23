/* A case where the validity of the barrier inside the if depends on
   the local size. If it's more than the constant in the predicate,
   the work-group's execution results will become undefined. */
__kernel void
test_kernel (void)
{
  unsigned local_id = get_local_id (0);

  printf ("LOCAL_ID=%d before if\n", local_id);
  if (local_id < 17)
  {
      printf ("LOCAL_ID=%d inside if\n", local_id);
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < 17)
        return;
  }
  printf ("LOCAL_ID=%d after if\n", local_id);
}
