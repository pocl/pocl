// A partial unroll request whose count doesn't divide the trip count (4 for
// 16 work-items) is left alone, as the unrolled copies would keep exits
// partway through the body. Both work-group functions keep the request.
kernel void tree_sum(global int *out, global const int *in) {
  local int s[256];
  size_t l = get_local_id(0);
  s[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll 3
  for (size_t k = get_local_size(0) / 2; k > 0; k >>= 1) {
    if (l < k)
      s[l] += s[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0)
    out[get_group_id(0)] = s[0];
}
