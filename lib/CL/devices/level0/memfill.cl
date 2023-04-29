/*
clang -x cl -Xclang -cl-std=CL1.2 -D__OPENCL_C_VERSION__=120 -D__OPENCL_VERSION__=120 -Dcl_khr_byte_addressable_store -Dcl_khr_global_int32_base_atomics -Dcl_khr_global_int32_extended_atomics -Dcl_khr_local_int32_base_atomics -Dcl_khr_local_int32_extended_atomics -Dcl_khr_int64 -Dcl_khr_spir -Xclang -cl-ext=-all,+cl_khr_byte_addressable_store,+cl_khr_global_int32_base_atomics,+cl_khr_global_int32_extended_atomics,+cl_khr_local_int32_base_atomics,+cl_khr_local_int32_extended_atomics,+cl_khr_spir -D__ENDIAN_LITTLE__=1 -emit-llvm -Xclang -finclude-default-header -target spir64-unknown-unknown -o memfill.bc -c memfill.cl
llvm-spirv -o memfill.spv memfill.bc
xxd -i memfill.spv >memfill.h
*/

#define MEMFILL_KERNELS(TYPE, SIZE, NITEMS)                                    \
  typedef struct Pattern##SIZE {                                               \
    TYPE data[NITEMS];                                                         \
  } pattern##SIZE##_t;                                                         \
                                                                               \
  void __kernel memfill_##SIZE(global TYPE *mem,                               \
                               const pattern##SIZE##_t pattern) {              \
    size_t gid = get_global_id(0) * NITEMS;                                    \
    for (size_t i = 0; i < NITEMS; ++i) {                                      \
      mem[gid + i] = pattern.data[i];                                          \
    }                                                                          \
  }                                                                            \
                                                                               \
  void __kernel memfill_rect_##SIZE(global TYPE *mem, const uint row_pitch,    \
                                    const uint slice_pitch,                    \
                                    const pattern##SIZE##_t pattern) {         \
    size_t start = get_global_id(2) * slice_pitch +                            \
                   get_global_id(1) * row_pitch + get_global_id(0) * NITEMS;   \
    for (size_t i = 0; i < NITEMS; ++i) {                                      \
      mem[start + i] = pattern.data[i];                                        \
    }                                                                          \
  }

// ###############################################################

MEMFILL_KERNELS(uchar, 1, 1)

MEMFILL_KERNELS(ushort, 2, 1)

MEMFILL_KERNELS(uint, 4, 1)

MEMFILL_KERNELS(uint, 8, 2)

MEMFILL_KERNELS(uint, 16, 4)

MEMFILL_KERNELS(uint, 32, 8)

MEMFILL_KERNELS(uint, 64, 16)

MEMFILL_KERNELS(uint, 128, 32)
