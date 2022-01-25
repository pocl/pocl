#include "pocl_cl.h"
#include "pocl_export.h"

#include <vector>

enum BuiltinKernelId : uint16_t
{
  // CD = custom device, BI = built-in
  // 1D array byte copy, get_global_size(0) defines the size of data to copy
  // kernel prototype: pocl.copy(char *input, char *output)
  POCL_CDBI_COPY = 0,
  POCL_CDBI_ADD32 = 1,
  POCL_CDBI_MUL32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_LAST = 5,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
};

// An initialization wrapper for kernel argument metadatas.
struct BIArg : public pocl_argument_info
{
  BIArg (const char *TypeName, const char *Name, pocl_argument_type Type,
         cl_kernel_arg_address_qualifier ADQ = CL_KERNEL_ARG_ADDRESS_GLOBAL,
         cl_kernel_arg_access_qualifier ACQ = CL_KERNEL_ARG_ACCESS_NONE,
         cl_kernel_arg_type_qualifier TQ = CL_KERNEL_ARG_TYPE_NONE,
         size_t size = 0)
  {
    name = strdup (Name);
    address_qualifier = ADQ;
    access_qualifier = ACQ;
    type_qualifier = TQ;
    type_name = strdup (TypeName);
    type_size = size;
    type = Type;
  }

  ~BIArg ()
  {
    free (name);
    free (type_name);
  }
};

// An initialization wrapper for kernel metadatas.
// BIKD = Built-in Kernel Descriptor
struct BIKD : public pocl_kernel_metadata_t
{
  BIKD (BuiltinKernelId KernelId, const char *KernelName,
        const std::vector<pocl_argument_info> &ArgInfos);

  ~BIKD ()
  {
    delete[] arg_info;
    free (name);
  }

  BuiltinKernelId KernelId;
};


#define BIKERNELS POCL_CDBI_LAST
POCL_EXPORT extern BIKD BIDescriptors[BIKERNELS];

extern "C" {
POCL_EXPORT
int pocl_setup_builtin_metadata(cl_device_id device, cl_program program,
                                unsigned program_device_i);
}
