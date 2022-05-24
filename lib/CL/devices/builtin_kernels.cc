
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
#define PTR_ARG POCL_ARG_TYPE_POINTER
#define GLOBAL_AS CL_KERNEL_ARG_ADDRESS_GLOBAL

BIKD BIDescriptors[BIKERNELS] = {BIKD(POCL_CDBI_COPY, "pocl.copy",
                             {BIArg("char*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("char*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_ADD32, "pocl.add32",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_MUL32, "pocl.mul32",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)})};
