int printf(const char *restrict format, ...);

__kernel void
test_kernel (void)
{
  int gid_x = get_global_id (0);
  int gid_y = get_global_id (1);
  int gid_z = get_global_id (2);

  volatile int loopcount = 2;

  for (int i = 0; i < loopcount; ++i) {
      printf ("i:%d %d %d %d before barrier\n", i, gid_x, gid_y, gid_z);

      barrier(CLK_LOCAL_MEM_FENCE);

      printf ("i:%d %d %d %d after barrier\n", i, gid_x, gid_y, gid_z);

      if (gid_x == 0) continue;

      printf ("i:%d %d %d %d after latch 1\n", i, gid_x, gid_y, gid_z);

      if (gid_x == 1) continue;

      printf ("i:%d %d %d %d after latch 2\n", i, gid_x, gid_y, gid_z);

  }
}
