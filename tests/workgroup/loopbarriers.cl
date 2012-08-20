__kernel void
test_kernel (void)
{
  unsigned group_id = get_group_id (0);
  unsigned local_id = get_local_id (0);

  for (volatile int i = 0; i < 3; ++i)
    {
      printf ("[GROUP_ID=%d] iteration=%d, A_before_barrier, local_id=%d\n",
              group_id, i, local_id);
      
      barrier(CLK_LOCAL_MEM_FENCE);

      printf ("[GROUP_ID=%d] iteration=%d, B_after_barrier, local_id=%d\n",
              group_id, i, local_id);
    }
}
