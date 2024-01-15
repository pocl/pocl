
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "builtin_kernels.hh"

// Shortcut handles to make the descriptor list more compact.
#define READ_BUF                                                               \
  POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL,                         \
      CL_KERNEL_ARG_ACCESS_NONE,                                               \
      CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_RESTRICT
#define WRITE_BUF                                                              \
  POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL,                         \
      CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_RESTRICT
#define POD_ARG_ATTRS                                                          \
  POCL_ARG_TYPE_NONE, CL_KERNEL_ARG_ADDRESS_PRIVATE,                           \
      CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_NONE
#define POD_ARG(num_bits) POD_ARG_ATTRS, ((num_bits + 7u) / 8u)
#define POD_ARG_32b POD_ARG(32)
#define READ_PIPE                                                              \
  POCL_ARG_TYPE_PIPE, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, \
      CL_KERNEL_ARG_TYPE_NONE, 4
#define WRITE_PIPE                                                             \
  POCL_ARG_TYPE_PIPE, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, \
      CL_KERNEL_ARG_TYPE_NONE, 4

BIKD pocl_BIDescriptors[BIKERNELS] = {
    BIKD(POCL_CDBI_COPY_I8, "pocl.copy.i8",
         {BIArg("char*", "input", READ_BUF),
          BIArg("char*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_ADD_I32, "pocl.add.i32",
         {BIArg("int*", "input1", READ_BUF), BIArg("int*", "input2", READ_BUF),
          BIArg("int*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_MUL_I32, "pocl.mul.i32",
         {BIArg("int*", "input1", READ_BUF), BIArg("int*", "input2", READ_BUF),
          BIArg("int*", "output", WRITE_BUF)}),
    BIKD(
        POCL_CDBI_LEDBLINK, "pocl.ledblink",
        {BIArg("int*", "input1", READ_BUF), BIArg("int*", "input2", READ_BUF)}),
    BIKD(
        POCL_CDBI_COUNTRED, "pocl.countred",
        {BIArg("int*", "input", READ_BUF), BIArg("int*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_DNN_CONV2D_RELU_I8, "pocl.dnn.conv2d.relu.i8",
         {
             BIArg("char*", "input", READ_BUF),
             BIArg("char*", "weights", READ_BUF),
             BIArg("char*", "output", WRITE_BUF),
             BIArg("int*", "bias", READ_BUF),
             BIArg("int*", "scale", READ_BUF),
             BIArg("int*", "shift", READ_BUF),
             BIArg("char*", "zero_point", READ_BUF),
             BIArg("unsigned", "window_size_x", POD_ARG_32b),
             BIArg("unsigned", "window_size_y", POD_ARG_32b),
             BIArg("unsigned", "stride_x", POD_ARG_32b),
             BIArg("unsigned", "stride_y", POD_ARG_32b),
             BIArg("unsigned", "input_depth", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_SGEMM_LOCAL_F32, "pocl.sgemm.local.f32",
         {
             BIArg("float*", "A", READ_BUF),
             BIArg("float*", "B", READ_BUF),
             BIArg("float*", "C", WRITE_BUF),
             BIArg("unsigned", "M", POD_ARG_32b),
             BIArg("unsigned", "N", POD_ARG_32b),
             BIArg("unsigned", "K", POD_ARG_32b),
         },
         2 * 16 * 16 * 4 // local mem size, 2 float arrays 16x16
         ),
    BIKD(POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE,
         "pocl.sgemm.scale.tensor.f16f16f32",
         {
             BIArg("half*", "A", READ_BUF),
             BIArg("half*", "B", READ_BUF),
             BIArg("float*", "C", WRITE_BUF),
             BIArg("unsigned", "M", POD_ARG_32b),
             BIArg("unsigned", "N", POD_ARG_32b),
             BIArg("unsigned", "K", POD_ARG_32b),
             BIArg("float", "alpha", POD_ARG_32b),
             BIArg("float", "beta", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_SGEMM_TENSOR_F16F16F32, "pocl.sgemm.tensor.f16f16f32",
         {
             BIArg("half*", "A", READ_BUF),
             BIArg("half*", "B", READ_BUF),
             BIArg("float*", "C", WRITE_BUF),
             BIArg("unsigned", "M", POD_ARG_32b),
             BIArg("unsigned", "N", POD_ARG_32b),
             BIArg("unsigned", "K", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_ABS_F32, "pocl.abs.f32",
         {BIArg("float*", "input", READ_BUF),
          BIArg("float*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_DNN_DENSE_RELU_I8, "pocl.dnn.dense.relu.i8",
         {
             BIArg("char*", "input", READ_BUF),
             BIArg("char*", "weights", READ_BUF),
             BIArg("char*", "output", WRITE_BUF),
             BIArg("int*", "bias", READ_BUF),
             BIArg("unsigned", "scale", POD_ARG_32b),
             BIArg("unsigned", "shift", POD_ARG_32b),
             BIArg("unsigned", "zero_point", POD_ARG_32b),
             BIArg("unsigned", "output_minus", POD_ARG_32b),
             BIArg("unsigned", "input_size", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_MAXPOOL_I8, "pocl.maxpool.i8",
         {
             BIArg("char*", "input", READ_BUF),
             BIArg("char*", "output", WRITE_BUF),
             BIArg("unsigned", "window_size_x", POD_ARG_32b),
             BIArg("unsigned", "window_size_y", POD_ARG_32b),
             BIArg("unsigned", "stride_x", POD_ARG_32b),
             BIArg("unsigned", "stride_y", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_ADD_I8, "pocl.add.i8",
         {BIArg("char*", "input1", READ_BUF),
          BIArg("char*", "input2", READ_BUF),
          BIArg("char*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_MUL_I8, "pocl.mul.i8",
         {BIArg("char*", "input1", READ_BUF),
          BIArg("char*", "input2", READ_BUF),
          BIArg("char*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_ADD_I16, "pocl.add.i16",
         {BIArg("short*", "input1", READ_BUF),
          BIArg("short*", "input2", READ_BUF),
          BIArg("short*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_MUL_I16, "pocl.mul.i16",
         {BIArg("short*", "input1", READ_BUF),
          BIArg("short*", "input2", READ_BUF),
          BIArg("short*", "output", WRITE_BUF)}),
    BIKD(POCL_CDBI_STREAMOUT_I32, "pocl.streamout.i32",
         {
             BIArg("int*", "output", READ_BUF),
         }),
    BIKD(POCL_CDBI_STREAMIN_I32, "pocl.streamin.i32",
         {
             BIArg("int*", "output", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_VOTE_U32, "pocl.vote.u32",
         {BIArg("int*", "output", WRITE_BUF),
          BIArg("unsigned", "num_inputs", POD_ARG_32b),
          BIArg("int*", "input0", READ_BUF),
          BIArg("int*", "input1", READ_BUF),
          BIArg("int*", "input2", READ_BUF),
          BIArg("int*", "input3", READ_BUF),
          BIArg("int*", "input4", READ_BUF),
          BIArg("int*", "input5", READ_BUF),
          BIArg("int*", "input6", READ_BUF),
          BIArg("int*", "input7", READ_BUF)
          }),
    BIKD(POCL_CDBI_VOTE_U8, "pocl.vote.u8",
         {BIArg("char*", "output", WRITE_BUF),
          BIArg("unsigned", "num_inputs", POD_ARG_32b),
          BIArg("char*", "input0", READ_BUF),
          BIArg("char*", "input1", READ_BUF),
          BIArg("char*", "input2", READ_BUF),
          BIArg("char*", "input3", READ_BUF),
          BIArg("char*", "input4", READ_BUF),
          BIArg("char*", "input5", READ_BUF),
          BIArg("char*", "input6", READ_BUF),
          BIArg("char*", "input7", READ_BUF)
          }),
   BIKD(POCL_CDBI_DNN_CONV2D_NCHW_F32, "pocl.dnn.conv2d.nchw.f32",
         {
             BIArg("float*", "input", READ_BUF),
             BIArg("float*", "weights", READ_BUF),
             BIArg("float*", "output", WRITE_BUF),
             BIArg("int", "input_n", POD_ARG_32b),
             BIArg("int", "input_c", POD_ARG_32b),
             BIArg("int", "input_h", POD_ARG_32b),
             BIArg("int", "input_w", POD_ARG_32b),
             BIArg("int", "filt_k", POD_ARG_32b),
             BIArg("int", "filt_c", POD_ARG_32b),
             BIArg("int", "filt_h", POD_ARG_32b),
             BIArg("int", "filt_w", POD_ARG_32b),
             BIArg("int", "stride_h", POD_ARG_32b),
             BIArg("int", "stride_w", POD_ARG_32b),
             BIArg("int", "dilation_h", POD_ARG_32b),
             BIArg("int", "dilation_w", POD_ARG_32b),
             BIArg("int", "padding_h", POD_ARG_32b),
             BIArg("int", "padding_w", POD_ARG_32b),
             BIArg("int", "groups", POD_ARG_32b),
             BIArg("float", "alpha", POD_ARG_32b),
             BIArg("float", "beta", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8,
         "org.khronos.openvx.scale_image.nn.u8",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("unsigned char*", "output", WRITE_BUF),
             BIArg("float", "width_scale", POD_ARG_32b),
             BIArg("float", "height_scale", POD_ARG_32b),
             BIArg("int", "input_width", POD_ARG_32b),
             BIArg("int", "input_height", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8,
         "org.khronos.openvx.scale_image.bl.u8",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("unsigned char*", "output", WRITE_BUF),
             BIArg("float", "width_scale", POD_ARG_32b),
             BIArg("float", "height_scale", POD_ARG_32b),
             BIArg("int", "input_width", POD_ARG_32b),
             BIArg("int", "input_height", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32,
         "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("float*", "output", WRITE_BUF),
             BIArg("float", "norm", POD_ARG_32b),
             BIArg("float", "offset", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_OPENVX_MINMAXLOC_R1_U8,
         "org.khronos.openvx.minmaxloc.r1.u8",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("unsigned char*", "min", WRITE_BUF),
             BIArg("unsigned char*", "max", WRITE_BUF),
             BIArg("unsigned int*", "minloc", WRITE_BUF),
             BIArg("unsigned int*", "maxloc", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_SOBEL3X3_U8,
         "pocl.sobel3x3.u8",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("unsigned short*", "sobel_x", WRITE_BUF),
             BIArg("unsigned short*", "sobel_y", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_PHASE_U8,
         "pocl.phase.u8",
         {
             BIArg("unsigned short*", "in_x", READ_BUF),
             BIArg("unsigned short*", "in_y", READ_BUF),
             BIArg("unsigned char*", "output", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_MAGNITUDE_U16,
         "pocl.magnitude.u16",
         {
             BIArg("unsigned short*", "in_x", READ_BUF),
             BIArg("unsigned short*", "in_y", READ_BUF),
             BIArg("unsigned short*", "output", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_ORIENTED_NONMAX_U16,
         "pocl.oriented.nonmaxsuppression.u16",
         {
             BIArg("unsigned short*", "magnitude", READ_BUF),
             BIArg("unsigned char*", "phase", READ_BUF),
             BIArg("unsigned char*", "output", WRITE_BUF),
             BIArg("unsigned short", "threshold_lower", POD_ARG_32b),
             BIArg("unsigned short", "threshold_upper", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_CANNY_U8,
         "pocl.canny.u8",
         {
             BIArg("unsigned char*", "input", READ_BUF),
             BIArg("unsigned char*", "output", WRITE_BUF),
             BIArg("unsigned short", "threshold_lower", POD_ARG_32b),
             BIArg("unsigned short", "threshold_upper", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_STREAM_MM2S_P512,
         "pocl.stream.mm2s.p512",
         {
             BIArg("char*", "in", READ_BUF),
             BIArg("uchar64", "out", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_STREAM_S2MM_P512,
         "pocl.stream.s2mm.p512",
         {
             BIArg("uchar64", "in", READ_PIPE),
             BIArg("char*", "out", WRITE_BUF),
         }),
    BIKD(POCL_CDBI_BROADCAST_1TO2_P512,
         "pocl.broadcast.1to2.p512",
         {
             BIArg("uchar64", "in", READ_PIPE),
             BIArg("uchar64", "out0", WRITE_PIPE),
             BIArg("uchar64", "out1", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_SOBEL3X3_P512,
         "pocl.sobel3x3.p512",
         {
             BIArg("uchar64", "in", READ_PIPE),
             BIArg("short32", "sobel_x", WRITE_PIPE),
             BIArg("short32", "sobel_y", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_PHASE_P512,
         "pocl.phase.p512",
         {
             BIArg("short32", "in_x", READ_PIPE),
             BIArg("short32", "in_y", READ_PIPE),
             BIArg("uchar64", "output", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_MAGNITUDE_P512,
         "pocl.magnitude.p512",
         {
             BIArg("short32", "in_x", READ_PIPE),
             BIArg("short32", "in_y", READ_PIPE),
             BIArg("ushort32", "output", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_ORIENTED_NONMAX_P512,
         "pocl.oriented.nonmaxsuppression.p512",
         {
             BIArg("ushort32", "magnitude", READ_PIPE),
             BIArg("uchar64", "phase", READ_PIPE),
             BIArg("uchar64", "output", WRITE_PIPE),
             BIArg("unsigned short", "threshold_lower", POD_ARG_32b),
             BIArg("unsigned short", "threshold_upper", POD_ARG_32b),
         }),
    BIKD(POCL_CDBI_GAUSSIAN3X3_P512,
         "pocl.gaussian3x3.p512",
         {
             BIArg("uchar64", "in",  READ_PIPE),
             BIArg("uchar64", "out", WRITE_PIPE),
         }),
    BIKD(POCL_CDBI_DBK_KHR_GEMM,
         "khr_gemm",
         {
           // The types are placeholders
           BIArg("unsigned char*", "a", READ_BUF),
           BIArg("unsigned char*", "b", READ_BUF),
           BIArg("unsigned char*", "c_in", READ_BUF),
           BIArg("unsigned char*", "c_out", WRITE_BUF),
         },
         0,
         /* isa_dbk= */ true),
    BIKD(POCL_CDBI_DBK_KHR_MATMUL,
         "khr_matmul",
         {
           BIArg("unsigned char*", "a", READ_BUF),
           BIArg("unsigned char*", "b", READ_BUF),
           BIArg("unsigned char*", "c", WRITE_BUF),
         },
         0,
         /* isa_dbk= */ true),
};

BIKD::BIKD(BuiltinKernelId KernelIdentifier, const char *KernelName,
           const std::vector<pocl_argument_info> &ArgInfos,
           unsigned local_mem_size, bool isa_dbk)
    : KernelId(KernelIdentifier) {

  builtin_kernel = isa_dbk ? POCL_DBK : POCL_BIK;
  builtin_max_global_work = {0, 0, 0};
  name = strdup(KernelName);
  num_args = ArgInfos.size();
  arg_info = new pocl_argument_info[num_args];
  int i = 0;
  data = NULL;

  if (local_mem_size > 0) {
    num_locals = 1;
    local_sizes = new size_t[1];
    local_sizes[0] = local_mem_size;
  } else
    num_locals = 0;

  for (auto ArgInfo : ArgInfos) {
    arg_info[i] = ArgInfo;
    arg_info[i].name = strdup(ArgInfo.name);
    arg_info[i].type_name = strdup(ArgInfo.type_name);
    ++i;
  }
}

static cl_int pocl_get_builtin_kernel_metadata(cl_device_id dev,
                                               const char *kernel_name,
                                               pocl_kernel_metadata_t *target) {

  BIKD *Desc = nullptr;
  for (size_t i = 0; i < BIKERNELS; ++i) {
    Desc = &pocl_BIDescriptors[i];
    if (std::string(Desc->name) == kernel_name) {
      memcpy(target, (pocl_kernel_metadata_t *)Desc,
             sizeof(pocl_kernel_metadata_t));
      target->name = strdup(Desc->name);
      target->arg_info = (struct pocl_argument_info *)calloc(
          Desc->num_args, sizeof(struct pocl_argument_info));
      memset(target->arg_info, 0,
             sizeof(struct pocl_argument_info) * Desc->num_args);
      for (unsigned Arg = 0; Arg < Desc->num_args; ++Arg) {
        memcpy(&target->arg_info[Arg], &Desc->arg_info[Arg],
               sizeof(pocl_argument_info));
        target->arg_info[Arg].name = strdup(Desc->arg_info[Arg].name);
        target->arg_info[Arg].type_name = strdup(Desc->arg_info[Arg].type_name);
        if (target->arg_info[Arg].type == POCL_ARG_TYPE_POINTER ||
            target->arg_info[Arg].type == POCL_ARG_TYPE_IMAGE)
          target->arg_info[Arg].type_size = sizeof (cl_mem);
      }
      target->builtin_max_global_work = Desc->builtin_max_global_work;
      target->has_arg_metadata =
        POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
        POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER  |
        POCL_HAS_KERNEL_ARG_TYPE_NAME         |
        POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER    |
        POCL_HAS_KERNEL_ARG_NAME;
    }
  }
  return 0;
}

int pocl_setup_builtin_metadata(cl_device_id device, cl_program program,
                                unsigned program_device_i) {
  if (program->builtin_kernel_names == nullptr)
    return 0;

  program->num_kernels = program->num_builtin_kernels;
  if (program->num_kernels) {
    program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
        program->num_kernels, sizeof(pocl_kernel_metadata_t));

    for (size_t i = 0; i < program->num_kernels; ++i) {
      pocl_get_builtin_kernel_metadata(device,
                                       program->builtin_kernel_names[i],
                                       &program->kernel_meta[i]);
      program->kernel_meta[i].data =
          (void**)calloc(program->num_devices, sizeof(void*));
    }
  }

  return 1;
}

int pocl_sanitize_builtin_kernel_name(cl_kernel kernel, char **saved_name) {
  *saved_name = nullptr;
  if (kernel->program->num_builtin_kernels) {
    *saved_name = kernel->meta->name;
    std::string name(kernel->name);
    for (BIKD &BI : pocl_BIDescriptors) {
      if (name.compare(BI.name) == 0) {
        std::replace(name.begin(), name.end(), '.', '_');
        kernel->meta->name = strdup(name.c_str());
        kernel->name = kernel->meta->name;
        break;
      }
    }
  }
  return 0;
}

int pocl_restore_builtin_kernel_name(cl_kernel kernel, char *saved_name) {
  if (kernel->program->num_builtin_kernels) {
    std::free((void *)kernel->name);
    kernel->meta->name = saved_name;
    kernel->name = kernel->meta->name;
  }
  return 0;
}
