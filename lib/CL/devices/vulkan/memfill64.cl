/* compile with:
clspv -x=cl --spv-version=1.0 --cl-kernel-arg-info --keep-unused-arguments --uniform-workgroup-size --global-offset --global-offset-push-constant  --cl-std=CL1.2  --cluster-pod-kernel-args  -o memfill64.spv memfill64.cl
*/

typedef struct pattern64 {
  uint data[16];
} pattern64_t;

void __kernel fill_64_mem (global uint *mem, const pattern64_t pattern)
{
  size_t gid = get_global_id(0) * 16;
  for (size_t i = 0; i < 16; ++i)
  {
    mem[gid + i] = pattern.data[i];
  }
}
