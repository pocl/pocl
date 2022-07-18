/* compile with:
clspv -x=cl --spv-version=1.0 --cl-kernel-arg-info --keep-unused-arguments --uniform-workgroup-size --global-offset --global-offset-push-constant  --cl-std=CL1.2  --cluster-pod-kernel-args  -o memfill128.spv memfill128.cl
*/
/* pattern = 128 bytes = 32 uints */

void __kernel fill_128_mem (global uint *mem, global uint *pattern)
{
  size_t gid = get_global_id(0) * 32;
  for (size_t i = 0; i < 32; ++i)
  {
    mem[gid + i] = pattern[i];
  }
}
