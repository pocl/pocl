// A tree reduction over the local size is fully unrolled in a work-group
// function specialized for the local size, as the size is folded before the
// barrier passes.
kernel void tree_sum(global int *out, global const int *in) {
  local int s[256];
  size_t l = get_local_id(0);
  s[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (size_t k = get_local_size(0) / 2; k > 0; k >>= 1) {
    if (l < k)
      s[l] += s[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0)
    out[get_group_id(0)] = s[0];
}
