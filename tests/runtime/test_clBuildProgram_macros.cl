#ifdef TEST_DOUBLES
#ifdef cl_khr_fp64
#  pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#  error "cl_khr_fp64 macro undefined"
#endif
#endif

#ifdef cl_khr_global_int32_base_atomics
#  pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : disable
#else
#  error "cl_khr_global_int32_base_atomics macro undefined"
#endif

__kernel void kernel_1() {
  printf("Hello World\n");
}
