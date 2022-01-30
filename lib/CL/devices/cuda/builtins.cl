__kernel void pocl_abs_f32(global const float* __restrict input,
                         global float* __restrict output)
{
  size_t i = get_global_id(0);
  output[i] = fabs(input[i]);
}
