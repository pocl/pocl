// A sub-group butterfly over a sub-group size that is known at compile time is
// fully unrolled before the barrier passes, so it doesn't become a b-loop.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__attribute__((intel_reqd_sub_group_size(8)))
kernel void sub_group_sum(global int *out, global const int *in) {
  size_t i = get_global_id(0);
  int v = in[i];
  for (uint m = 1; m < get_max_sub_group_size(); m <<= 1)
    v += sub_group_shuffle_xor(v, m);
  out[i] = v;
}
