// Test the different ID functions. Mainly to check Range MD generation, which
// has to be verified manually in the parallel.bc currently.
__kernel void
test_kernel (global int *output)
{
  if (get_global_id (0) == 0)
    {
      output[0] = get_local_size (0);
      output[1] = get_local_size (1);
      output[2] = get_local_size (2);
      output[3] = get_work_dim ();
      output[4] = get_num_groups (0);
      output[5] = get_num_groups (1);
      output[6] = get_num_groups (2);
      output[7] = get_global_offset (0);
      output[8] = get_global_offset (1);
      output[9] = get_global_offset (2);
    }
  else if (get_global_id (0) > 9)
    output[get_global_id (0)] = 0;
}
