// A short loop carrying a wide value across its barriers is fully unrolled
// without an unroll request only while the context arrays of the copies fit
// the stack budget: with a local size of 256, but not 4096.
kernel void wide(global uint16 *out, global const uint16 *in) {
  local uint s[4096];
  size_t l = get_local_id(0), i = get_global_id(0);
  uint16 acc = in[i];
  for (int it = 0; it < 16; it++) {
    s[l] = acc.s0;
    barrier(CLK_LOCAL_MEM_FENCE);
    acc = acc * 3 + s[(l + 1) & (get_local_size(0) - 1)];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[i] = acc;
}
