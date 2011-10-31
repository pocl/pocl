int printf(const char *restrict format, ...);

__kernel void
loopbarriers (void)
{
  int gid = get_global_id (0);

  for (int i = 0; i < 2; ++i)
    {
      printf ("%d: iteration %d, before barrier()\n", gid, i);
      
      barrier(CLK_LOCAL_MEM_FENCE);
      
      printf ("%d: iteration %d, after barrier()\n", gid, i);
    }
}
