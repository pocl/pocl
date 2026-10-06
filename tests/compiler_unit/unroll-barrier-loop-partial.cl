// A partial unroll request on a tree reduction over the local size would
// leave a b-loop, so the loop isn't unrolled before the barrier passes, even
// though the trip count is a constant in a work-group function specialized
// for the size. It stays a b-loop, which keeps its iteration variable per
// work-item in a context array.
kernel void tree_sum(global int *out, global const int *in) {
  local int s[256];
  size_t l = get_local_id(0);
  s[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll 2
  for (size_t k = get_local_size(0) / 2; k > 0; k >>= 1) {
    if (l < k)
      s[l] += s[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0)
    out[get_group_id(0)] = s[0];
}
