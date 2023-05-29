/*
clang -x cl -Xclang -cl-std=CL1.2 -D__OPENCL_C_VERSION__=120 -D__OPENCL_VERSION__=120 -Dcl_khr_byte_addressable_store -Dcl_khr_global_int32_base_atomics -Dcl_khr_global_int32_extended_atomics -Dcl_khr_local_int32_base_atomics -Dcl_khr_local_int32_extended_atomics -Dcl_khr_int64 -Dcl_khr_spir -D cl_khr_3d_image_writes -Xclang -cl-ext=-all,+cl_khr_byte_addressable_store,+cl_khr_global_int32_base_atomics,+cl_khr_global_int32_extended_atomics,+cl_khr_local_int32_base_atomics,+cl_khr_local_int32_extended_atomics,+cl_khr_spir,+cl_khr_3d_image_writes -D__ENDIAN_LITTLE__=1 -emit-llvm -Xclang -finclude-default-header -target spir64-unknown-unknown -o imagefill.bc -c imagefill.cl
llvm-spirv --spirv-max-version=1.2 -o imagefill.spv imagefill.bc
xxd -i imagefill.spv >imagefill.h
*/

#define IMAGEFILL_KERNELS(SUFFIX, PIX_TYPE, DEPTH_TYPE)                       \
                                                                              \
  void __kernel imagefill_2d_##SUFFIX (write_only image2d_t img,              \
                                       const PIX_TYPE color)                  \
  {                                                                           \
    int2 coord = (int2)(get_global_id (0), get_global_id (1));                \
    write_image##SUFFIX (img, coord, color);                                  \
  }                                                                           \
                                                                              \
  void __kernel imagefill_2d_array_##SUFFIX (write_only image2d_array_t img,  \
                                             const PIX_TYPE color)            \
  {                                                                           \
    int4 coord                                                                \
        = (int4)(get_global_id (0), get_global_id (1), get_global_id (2), 0); \
    write_image##SUFFIX (img, coord, color);                                  \
  }                                                                           \
                                                                              \
  void __kernel imagefill_1d_##SUFFIX (write_only image1d_t img,              \
                                       const PIX_TYPE color)                  \
  {                                                                           \
    int coord = get_global_id (0);                                            \
    write_image##SUFFIX (img, coord, color);                                  \
  }                                                                           \
  void __kernel imagefill_1d_buffer_##SUFFIX (                                \
      write_only image1d_buffer_t img, const PIX_TYPE color)                  \
  {                                                                           \
    int coord = get_global_id (0);                                            \
    write_image##SUFFIX (img, coord, color);                                  \
  }                                                                           \
                                                                              \
  void __kernel imagefill_1d_array_##SUFFIX (write_only image1d_array_t img,  \
                                             const PIX_TYPE color)            \
  {                                                                           \
    int2 coord = (int2)(get_global_id (0), get_global_id (1));                \
    write_image##SUFFIX (img, coord, color);                                  \
  }                                                                           \
                                                                              \
  void __kernel imagefill_3d_##SUFFIX (write_only image3d_t img,              \
                                       const PIX_TYPE color)                  \
  {                                                                           \
    int4 coord                                                                \
        = (int4)(get_global_id (0), get_global_id (1), get_global_id (2), 0); \
    write_image##SUFFIX (img, coord, color);                                  \
  }

// ###############################################################

IMAGEFILL_KERNELS (f, float4, float)

IMAGEFILL_KERNELS (ui, uint4, uint)

IMAGEFILL_KERNELS (i, int4, int)

// ###############################################################

void __kernel
imagefill_2d_depth (write_only image2d_depth_t img, const float color)
{
  int2 coord = (int2)(get_global_id (0), get_global_id (1));
  write_imagef (img, coord, color);
}

void __kernel
imagefill_2d_array_depth (write_only image2d_array_depth_t img,
                          const float color)
{
  int4 coord
      = (int4)(get_global_id (0), get_global_id (1), get_global_id (2), 0);
  write_imagef (img, coord, color);
}
