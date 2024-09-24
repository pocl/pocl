
/* Workaround for the VPlan/LLVM crash reported here:
   https://github.com/pocl/pocl/issues/1556 */
#define volatile

kernel void
test_kernel (global int *output)
{
  size_t flat_id = get_global_id (2) * get_global_size (1)
                   + get_global_id (1) * get_global_size (0)
                   + get_global_id (0);

  size_t grid_size
      = get_global_size (2) * get_global_size (1) * get_global_size (0);

  for (volatile int i = 0; i < 3; ++i)
    {
      output[flat_id] = flat_id * 1000 + i;

      barrier (CLK_GLOBAL_MEM_FENCE);

      int temp = output[flat_id + 1 == grid_size ? 0 : (flat_id + 1)];

      barrier (CLK_GLOBAL_MEM_FENCE);
      /* If the barrier was ignored, we are likely copying
         a zero from the neighbour's slot or the previous
         value (in case the iterations are executed in
         lock step). */
      output[flat_id] = temp;
    }
}
