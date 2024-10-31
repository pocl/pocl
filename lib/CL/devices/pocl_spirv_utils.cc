/* pocl_spirv_utils.cc - a collection of functions useful when using SPIR-V
 * internally in PoCL.
 *
 * Copyright (c) 2022-2024 Michal Babej / Intel Finland Oy
 * Copyright (c) 2024 Robin Bijl / Tampere university
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "pocl_spirv_utils.hh"

using namespace SPIRVParser;

void mapToPoCLMetadata(OCLFuncInfo *funcInfo, const std::string& kernelName,
                       size_t numDevices,
                       pocl_kernel_metadata_t *kernelMetadata) {

  kernelMetadata->data = (void **)calloc(numDevices, sizeof(void *));
  kernelMetadata->num_args = funcInfo->ArgTypeInfo.size();
  kernelMetadata->name = strdup(kernelName.c_str());

  kernelMetadata->num_locals = 0;
  kernelMetadata->local_sizes = nullptr;

  kernelMetadata->max_subgroups = (size_t *)calloc(numDevices, sizeof(size_t));
  kernelMetadata->compile_subgroups =
      (size_t *)calloc(numDevices, sizeof(size_t));
  kernelMetadata->max_workgroup_size =
      (size_t *)calloc(numDevices, sizeof(size_t));
  kernelMetadata->preferred_wg_multiple =
      (size_t *)calloc(numDevices, sizeof(size_t));
  kernelMetadata->local_mem_size =
      (cl_ulong *)calloc(numDevices, sizeof(cl_ulong));
  kernelMetadata->private_mem_size =
      (cl_ulong *)calloc(numDevices, sizeof(cl_ulong));
  kernelMetadata->spill_mem_size =
      (cl_ulong *)calloc(numDevices, sizeof(cl_ulong));

  // ARGUMENTS
  if (kernelMetadata->num_args < 1)
    return;

  kernelMetadata->arg_info = (struct pocl_argument_info *)calloc(
      kernelMetadata->num_args, sizeof(struct pocl_argument_info));

  for (uint32_t J = 0; J < kernelMetadata->num_args; ++J) {
      cl_kernel_arg_address_qualifier Addr;
      cl_kernel_arg_access_qualifier Access;
      Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
      Access = CL_KERNEL_ARG_ACCESS_NONE;
      kernelMetadata->arg_info[J].name =
          strdup(funcInfo->ArgTypeInfo[J].Name.c_str());
      kernelMetadata->arg_info[J].type_name = nullptr;
      switch (funcInfo->ArgTypeInfo[J].Type) {
      case OCLType::POD: {
          kernelMetadata->arg_info[J].type = POCL_ARG_TYPE_NONE;
          kernelMetadata->arg_info[J].type_size = funcInfo->ArgTypeInfo[J].Size;
          break;
        }
      case OCLType::Pointer: {
          kernelMetadata->arg_info[J].type = POCL_ARG_TYPE_POINTER;
          kernelMetadata->arg_info[J].type_size = sizeof(cl_mem);
          switch (funcInfo->ArgTypeInfo[J].Space) {
          case OCLSpace::Private:
            Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
              break;
          case OCLSpace::Local:
            Addr = CL_KERNEL_ARG_ADDRESS_LOCAL;
              break;
          case OCLSpace::Global:
            Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
              break;
          case OCLSpace::Constant:
            Addr = CL_KERNEL_ARG_ADDRESS_CONSTANT;
              break;
          case OCLSpace::Unknown:
            Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
              break;
            }
          break;
        }
      case OCLType::Image: {
          kernelMetadata->arg_info[J].type = POCL_ARG_TYPE_IMAGE;
          kernelMetadata->arg_info[J].type_size = sizeof(cl_mem);
          Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
          bool Readable = funcInfo->ArgTypeInfo[J].Attrs.ReadableImg;
          bool Writable = funcInfo->ArgTypeInfo[J].Attrs.WriteableImg;
          if (Readable && Writable) {
              Access = CL_KERNEL_ARG_ACCESS_READ_WRITE;
            }
          if (Readable && !Writable) {
              Access = CL_KERNEL_ARG_ACCESS_READ_ONLY;
            }
          if (!Readable && Writable) {
              Access = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
            }
          break;
        }
      case OCLType::Sampler: {
          kernelMetadata->arg_info[J].type = POCL_ARG_TYPE_SAMPLER;
          kernelMetadata->arg_info[J].type_size = sizeof(cl_mem);
          break;
        }
      case OCLType::Opaque: {
          POCL_MSG_ERR("Unknown OCL type OPaque\n");
          kernelMetadata->arg_info[J].type = POCL_ARG_TYPE_NONE;
          kernelMetadata->arg_info[J].type_size = funcInfo->ArgTypeInfo[J].Size;
          break;
        }
        }
      kernelMetadata->arg_info[J].address_qualifier = Addr;
      kernelMetadata->arg_info[J].access_qualifier = Access;
      kernelMetadata->arg_info[J].type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
      if (funcInfo->ArgTypeInfo[J].Attrs.Constant) {
          kernelMetadata->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
        }
      if (funcInfo->ArgTypeInfo[J].Attrs.Restrict) {
          kernelMetadata->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
        }
      if (funcInfo->ArgTypeInfo[J].Attrs.Volatile) {
          kernelMetadata->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
        }
    }

  // TODO: POCL_HAS_KERNEL_ARG_TYPE_NAME missing
  kernelMetadata->has_arg_metadata = POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER |
                                     POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
                                     POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER |
                                     POCL_HAS_KERNEL_ARG_NAME;
}

void mapToPoCLMetadata(
    std::pair<const std::string, std::shared_ptr<OCLFuncInfo>> &pair,
size_t numDevices, pocl_kernel_metadata_t *kernelMetadata) {
mapToPoCLMetadata(pair.second.get(), pair.first, numDevices, kernelMetadata);
}