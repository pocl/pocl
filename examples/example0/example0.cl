__kernel void
integer_mad (__global const uint *a, __global const uint *b, __global uint *c)
{
  int gid = get_global_id (0);

  uint prod = a[gid] * 7 + b[gid];
  c[gid] = prod;
}
