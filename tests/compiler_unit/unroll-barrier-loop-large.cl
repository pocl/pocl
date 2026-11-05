// A loop with barriers and a constant trip count stays a b-loop without an
// unroll request if fully unrolling it exceeds LLVM's full unroll threshold,
// which LoopBarriers marks with a barrier in the preheader.
kernel void rotate_local(global int *out, global const int *in) {
  local int s[64];
  size_t l = get_local_id(0);
  int v = in[get_global_id(0)];
  for (uint k = 0; k < 64; ++k) {
    s[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    v = s[(l + 1) % 64];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[get_global_id(0)] = v;
}
