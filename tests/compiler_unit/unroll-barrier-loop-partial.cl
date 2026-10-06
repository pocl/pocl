// A partial unroll request whose count divides the trip count is applied
// before the barrier passes in a work-group function specialized for the local
// size (the trip count is 4 for 16 work-items), leaving a b-loop with two
// copies of the body. The generic function keeps the request for stage 2.
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
