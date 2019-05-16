
kernel void
test_kernel (global int *output)
{
  int gid = get_global_id (0);

  size_t flat_id = get_global_id (2) * get_global_size (1)
                   + get_global_id (1) * get_global_size (0)
                   + get_global_id (0);

  output[flat_id] = 0;

  /* Volatile operand here so LLVM doesn't optimize the loop away. */
  volatile int add = 1;

  for (int i = 0; i < gid; ++i)
    output[flat_id] += add;
}
