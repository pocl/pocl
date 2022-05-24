
#include <vector>

#include "builtin_kernels.hh"

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

// Shortcut handles to make the descriptor list more compact.
#define READ_BUF POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_CONST
#define WRITE_BUF POCL_ARG_TYPE_POINTER, CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_NONE

BIKD BIDescriptors[BIKERNELS] = {BIKD(POCL_CDBI_COPY, "pocl.copy",
                             {BIArg("char*", "input", READ_BUF),
                              BIArg("char*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_ADD32, "pocl.add32",
                             {BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_MUL32, "pocl.mul32",
                             {BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)}),
                        BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
                             {BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "input", READ_BUF)}),
                        BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
                             {BIArg("int*", "input", READ_BUF),
                              BIArg("int*", "output", WRITE_BUF)})};
