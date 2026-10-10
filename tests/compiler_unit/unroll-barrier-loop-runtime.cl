// A butterfly bounded by a kernel argument stays a b-loop, which LoopBarriers
// marks with a barrier in the preheader.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
kernel void sub_group_sum(global int *out, global const int *in, uint width) {
  size_t i = get_global_id(0);
  int v = in[i];
  for (uint m = 1; m < width; m <<= 1)
    v += sub_group_shuffle_xor(v, m);
  out[i] = v;
}
