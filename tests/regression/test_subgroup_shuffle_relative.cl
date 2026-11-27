/* Kernel for test_subgroup_shuffle_relative.c, which builds it from source
   and also loads test_subgroup_shuffle_relative.spv, built from this file
   with LLVM 23 and the SPIRV-LLVM-Translator:

     clang -cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 \
       -cl-ext=-cl_khr_fp64,-__opencl_c_fp64,-cl_khr_fp16 \
       -fdeclare-opencl-builtins -finclude-default-header \
       -emit-llvm-bc -O0 -disable-llvm-passes \
       test_subgroup_shuffle_relative.cl -o test_subgroup_shuffle_relative.bc
     llvm-spirv --spirv-max-version=1.3 test_subgroup_shuffle_relative.bc \
       -o test_subgroup_shuffle_relative.spv

   The SPIR-V module leaves out double and half, so that it loads on devices
   without them. */

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

/* Each type gets 8 slots per work-item: shuffle_up with the 4 deltas, then
   shuffle_down with them. The last delta varies across the sub-group. */
#define TEST(IDX, TYPE, VAL)                                                  \
  {                                                                           \
    TYPE v = (TYPE)(VAL);                                                     \
    for (int i = 0; i < 4; i++)                                               \
      {                                                                       \
        out[((IDX) * 8 + i) * n + gid]                                        \
          = (long)sub_group_shuffle_up (v, d[i]);                             \
        out[((IDX) * 8 + 4 + i) * n + gid]                                    \
          = (long)sub_group_shuffle_down (v, d[i]);                           \
      }                                                                       \
  }

/* The host defines REQD_SG_SIZE to get partial sub-groups. */
#ifdef REQD_SG_SIZE
__attribute__ ((intel_reqd_sub_group_size (REQD_SG_SIZE)))
#endif
kernel void
shuffle_relative (global uint *info, global long *out)
{
  uint gid = get_global_id (0);
  uint n = get_global_size (0);
  uint lid = get_sub_group_local_id ();
  uint sg_size = get_sub_group_size ();
  uint d[4] = { 0, 1, sg_size - 1, lid % 3 };

  info[2 * gid] = sg_size;
  info[2 * gid + 1] = lid;

  TEST (0, char, gid + 1)
  TEST (1, ushort, gid + 2)
  TEST (2, int, gid + 3)
  TEST (3, ulong, ((ulong)gid << 33) + 4)
  TEST (4, float, gid + 5)
#ifdef cl_khr_fp64
  TEST (5, double, gid + 6)
#endif
#ifdef cl_khr_fp16
  TEST (6, half, gid + 7)
#endif
}
