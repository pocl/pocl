int printf(const char *restrict format, ...);

__kernel void
barriers (void)
{
  int gid = get_global_id (0);

  printf ("%d: before barriers\n", gid);

  barrier(CLK_LOCAL_MEM_FENCE);

  printf ("%d: between barriers\n", gid);

  barrier(CLK_LOCAL_MEM_FENCE);

  printf ("%d: after barriers\n", gid);
}
