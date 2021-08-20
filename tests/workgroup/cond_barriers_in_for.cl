__kernel void
test_kernel (global int *output)
{
  __local int scratch[256]; // max wg size: 256

  const int gid_x = get_global_id (0);
  const int gid_y = get_global_id (1);
  const int gid_z = get_global_id (2);

  const int global_id = gid_x + get_global_size(0) * (gid_y + get_global_size(1) * gid_z);
  const int local_id = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
  int group_id = global_id / (get_local_size(0) * get_local_size(1) * get_local_size(2));

  scratch[local_id] = global_id;

  
  for (int i = 0; i < 2; i++)
  {
    if (group_id > 0)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(group_id == 1)
      {
        const int v = scratch[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[local_id] += v;
      }
    }
    else
    {
      scratch[local_id] += 1;
      barrier(CLK_LOCAL_MEM_FENCE);
      scratch[local_id] += 1;
    }
  }
  output[global_id] = scratch[local_id];
}
