int printf(const char *restrict format, ...);

__kernel void
test_kernel (void)
{
  int gid_x = get_global_id (0);
  int gid_y = get_global_id (1);
  int gid_z = get_global_id (2);

  printf ("%d %d %d: before barriers\n", gid_x, gid_y, gid_z);

  barrier(CLK_LOCAL_MEM_FENCE);

  printf ("%d %d %d: between barriers\n", gid_x, gid_y, gid_z);

  barrier(CLK_LOCAL_MEM_FENCE);

  printf ("%d %d %d: after barriers\n", gid_x, gid_y, gid_z);
}
