
#include <vector>
#include <string>

#include "builtin_kernels.hh"

// Shortcut handles to make the descriptor list more compact.
#define READ_BUF POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_RESTRICT
#define WRITE_BUF POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_RESTRICT

BIKD BIDescriptors[BIKERNELS] = {BIKD(POCL_CDBI_COPY, "pocl.copy",
                             {BIArg("char*", "input", READ_BUF),
                              BIArg("char*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_ADD32, "pocl.add32",
                             {BIArg("int*", "input1", READ_BUF),
                              BIArg("int*", "input2", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_MUL32, "pocl.mul32",
                             {BIArg("int*", "input1", READ_BUF),
                              BIArg("int*", "input2", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
                             {BIArg("int*", "input1", READ_BUF),
                              BIArg("int*", "input2", READ_BUF)}),
                        BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
                             {BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_DNN_CONV2D_INT8_RELU,
                              "pocl.dnn.conv2d_int8_relu",
                             {BIArg("char*", "placeholder", WRITE_BUF),
                              BIArg("char*", "placeholder1", WRITE_BUF),
                              BIArg("char*", "compute", WRITE_BUF),
                              BIArg("int*", "placeholder2", WRITE_BUF),
                              BIArg("int*", "placeholder3", WRITE_BUF),
                              BIArg("int*", "placeholder4", WRITE_BUF),
                              BIArg("int*", "placeholder5", WRITE_BUF),
                              BIArg("int*", "placeholder6", WRITE_BUF),
                              }),
                              };

BIKD::BIKD(BuiltinKernelId KernelIdentifier, const char *KernelName,
           const std::vector<pocl_argument_info> &ArgInfos)
    : KernelId(KernelIdentifier) {

  builtin_kernel = 1;
  name = strdup(KernelName);
  num_args = ArgInfos.size();
  arg_info = new pocl_argument_info[num_args];
  int i = 0;
  for (auto ArgInfo : ArgInfos) {
    arg_info[i] = ArgInfo;
    arg_info[i].name = strdup(ArgInfo.name);
    arg_info[i].type_name = strdup(ArgInfo.type_name);
    ++i;
  }
}



static cl_int
pocl_get_builtin_kernel_metadata(cl_device_id dev,
                                 const char *kernel_name,
                                 pocl_kernel_metadata_t *target) {

  BIKD *Desc = nullptr;
  for (size_t i = 0; i < BIKERNELS;  ++i) {
    Desc = &BIDescriptors[i];
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
    }
  }

  return 1;
}
