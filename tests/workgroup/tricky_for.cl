__kernel void
test_kernel (global int *output)
{
  int gid_x = get_global_id (0);

  for (int i = 0; i < INT_MAX; ++i) {
      if (i == 1 && gid_x == 0) {
          output[gid_x] = i * 1000;
          break;
      }
      output[gid_x] = -1;
      if (i == 1 && gid_x == 1) {
          output[gid_x] = i * 2000;
          break;
      }
      /* None of the two WIs reach the barrier, it should be fine! */
      output[gid_x] = -2;
      barrier(CLK_LOCAL_MEM_FENCE);
      output[gid_x] = -3;
  }
}
