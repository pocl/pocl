__kernel void
pocl_mul_i32 (global const int *__restrict input1,
              global const int *__restrict input2,
              global int *__restrict output)
{
  size_t i = get_global_id (0);
  output[i] = input1[i] * input2[i];
}

__kernel void
pocl_add_i32 (global const int *__restrict input1,
              global const int *__restrict input2,
              global int *__restrict output)
{
  size_t i = get_global_id (0);
#ifdef cl_TCE_ADD
  clADDTCE (input1[i], input2[i], output[i]);
#else
  output[i] = input1[i] + input2[i];
#endif
}

__kernel void
pocl_abs_f32 (global const int *__restrict input,
              global int *__restrict output)
{
  size_t i = get_global_id (0);
#ifdef cl_TCE_ABSF
  clABSFTCE (input[i], output[i]);
#else
  output[i] = fabs (input[i]);
#endif
}
