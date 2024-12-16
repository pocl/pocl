__kernel void
test_kernel (global int *output)
{
  int gid_x = get_global_id (0);

  for (volatile int i = 0; i < 100; ++i)
    {
      if (i == 1 && gid_x == 0)
        {
          output[gid_x] = i * 1000;
          return;
        }
      output[gid_x] = -1;
      if (i == 1 && gid_x == 1)
        {
          output[gid_x] = i * 2000;
          return;
        }
      output[gid_x] = -2;
    }
  if (gid_x > 3)
    {
      output[gid_x] = 100;
    }
  else if (gid_x == 2)
    {
      output[gid_x] = 200;
    }
}
