int printf(const char *restrict format, ...);

__kernel void
test_kernel (void)
{
  int gid = get_global_id (0);

  printf ("%d: ", gid);
  for (int i = 0; i < gid; ++i)
    printf ("%d ", i);
  printf ("\n");
}
