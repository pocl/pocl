kernel void test_vector_width(__global float* a, __global float* b, __global float* c)
{
  size_t i = get_global_id(0);
  c[i] = a[i] + b[i];
}
