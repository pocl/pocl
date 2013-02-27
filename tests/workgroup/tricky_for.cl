int printf(const char *restrict format, ...);

__kernel void
test_kernel (void)
{
  int gid_x = get_global_id (0);
  int gid_y = get_global_id (1);
  int gid_z = get_global_id (2);

  for (int i = 0; i < INT_MAX; ++i) {
      if (i == 1 && gid_x == 0) {
          printf ("I am 0 and I break out from the loop.\n");
          break;
      }
      if (i == 1 && gid_x == 1) {
          printf ("I am 1 and I also break out from the loop.\n");
          break;
      }
      /* None of the two WIs reach the barrier, it should be fine! */
      barrier(CLK_LOCAL_MEM_FENCE);
      printf ("gid_x %u after barrier at iteration %d\n", gid_x, i);
  }
}
