// #define PRINT_PATH

__kernel void
test_kernel (global int *output)
{
  __local int scratch[256]; // max wg size: 256

  const int gid_x = get_global_id (0);
  const int gid_y = get_global_id (1);
  const int gid_z = get_global_id (2);

  const int global_id
      = gid_x + get_global_size (0) * (gid_y + get_global_size (1) * gid_z);
  const int local_id
      = get_local_id (0)
        + get_local_size (0)
              * (get_local_id (1) + get_local_size (1) * get_local_id (2));
  int group_id
      = global_id
        / (get_local_size (0) * get_local_size (1) * get_local_size (2));

  scratch[local_id] = global_id;

  for (int i = 0; i < 2; i++)
    {
#ifdef PRINT_PATH
      printf ("A i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
      if (group_id > 0)
        {
#ifdef PRINT_PATH
          printf ("B i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
          barrier (CLK_LOCAL_MEM_FENCE);
          if (group_id == 1)
            {
              const int v = scratch[0];
#ifdef PRINT_PATH
              printf ("C i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
              barrier (CLK_LOCAL_MEM_FENCE);
              scratch[local_id] += v;
            }
#ifdef PRINT_PATH
          printf ("D i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
        }
      else
        {
          scratch[local_id] += 1;
#ifdef PRINT_PATH
          printf ("E i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
          barrier (CLK_LOCAL_MEM_FENCE);
          scratch[local_id] += 1;
#ifdef PRINT_PATH
          printf ("F i %d gid_x %d group_id %d\n", i, gid_x, group_id);
#endif
        }
    }
  output[global_id] = scratch[local_id];
}
