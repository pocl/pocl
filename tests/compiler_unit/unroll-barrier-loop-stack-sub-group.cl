// At a local size of 4096, a sub-group butterfly over the sub-group size is
// still fully unrolled without an unroll request, but a loop whose shuffles
// each need a wide scratch buffer for the work-group is not, as every copy
// allocates its own. At 256 the scratch fits the budget.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
kernel void butterfly(global int *out, global const int *in) {
  size_t i = get_global_id(0);
  int v = in[i];
  for (uint m = 1; m < get_max_sub_group_size(); m <<= 1)
    v += sub_group_shuffle_xor(v, m);
  out[i] = v;
}

kernel void scratch(global uint *out, global const uint16 *in) {
  size_t i = get_global_id(0);
  uint lane = get_sub_group_local_id();
  uint16 v = in[i];
  uint acc = 0;
  for (uint m = 1; m < 256; m <<= 1) {
    uint16 w = intel_sub_group_shuffle(v, lane ^ m);
    acc += w.s0 + w.sf;
  }
  out[i] = acc;
}
