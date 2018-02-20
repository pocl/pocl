/* pocl_llvm_metadata.cc: part of pocl LLVM API dealing with kernel metadata.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2017 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <llvm/Support/Casting.h>
#include <llvm/Support/MutexGuard.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/ADT/SmallVector.h>

#include <string>
#include <map>
#include <iostream>
#include <sstream>

#include "pocl_cl.h"
#include "pocl_llvm_api.h"
#include "pocl_cache.h"
#include "LLVMUtils.h"

using namespace llvm;

// The old way of getting kernel metadata from "opencl.kernels" module meta.
// LLVM < 3.9 and SPIR
static int pocl_get_kernel_arg_module_metadata(const char *kernel_name,
                                               llvm::Module *input,
                                               cl_kernel kernel) {
  // find the right kernel in "opencl.kernels" metadata
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  llvm::MDNode *kernel_metadata = NULL;

#ifdef LLVM_OLDER_THAN_3_9
  assert(opencl_kernels && opencl_kernels->getNumOperands());
#else
  if (!(opencl_kernels && opencl_kernels->getNumOperands()))
    // Perhaps it is a SPIR kernel without the "opencl.kernels" metadata
    return 1;
#endif

  for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
    llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

    llvm::Value *meta =
        dyn_cast<llvm::ValueAsMetadata>(kernel_iter->getOperand(0))->getValue();
    llvm::Function *kernel_prototype = llvm::cast<llvm::Function>(meta);
    std::string name = kernel_prototype->getName().str();
    if (name == kernel_name) {
      kernel_metadata = kernel_iter;
      break;
    }
  }

  kernel->arg_info = (struct pocl_argument_info *)calloc(
      kernel->num_args, sizeof(struct pocl_argument_info));
  memset(kernel->arg_info, 0,
         sizeof(struct pocl_argument_info) * kernel->num_args);

  kernel->has_arg_metadata = 0;

  assert(kernel_metadata && "kernel NOT found in opencl.kernels metadata");

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  int BitcodeIsSPIR = input->getTargetTriple().find("spir") == 0;
#endif

  unsigned e = kernel_metadata->getNumOperands();
  for (unsigned i = 1; i != e; ++i) {
    llvm::MDNode *meta_node =
        llvm::cast<MDNode>(kernel_metadata->getOperand(i));

    // argument num
    unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
    int has_meta_for_every_arg = ((arg_num - 1) == kernel->num_args);
#endif

    llvm::MDString *meta_name_node =
        llvm::cast<MDString>(meta_node->getOperand(0));
    std::string meta_name = meta_name_node->getString().str();

    for (unsigned j = 1; j != arg_num; ++j) {
      llvm::Value *meta_arg_value = NULL;
      if (isa<ValueAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
            dyn_cast<ValueAsMetadata>(meta_node->getOperand(j))->getValue();
      else if (isa<ConstantAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
            dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
      struct pocl_argument_info *current_arg = &kernel->arg_info[j - 1];

      if (meta_arg_value != NULL && isa<ConstantInt>(meta_arg_value) &&
          meta_name == "kernel_arg_addr_space") {
        assert(has_meta_for_every_arg &&
               "kernel_arg_addr_space meta incomplete");
        kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
        // std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
        llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
        uint64_t val = m->getLimitedValue(UINT_MAX);
        bool SPIRAddressSpaceIDs;
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
        SPIRAddressSpaceIDs = BitcodeIsSPIR;
#else
        // We have an LLVM fixed to produce always SPIR AS ids for the argument
        // info metadata.
        SPIRAddressSpaceIDs = true;
#endif

        if (SPIRAddressSpaceIDs) {
          switch (val) {
          case 0:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            break;
          case 1:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
            break;
          case 3:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
            break;
          case 2:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT;
            break;
          }
        } else {
          switch (val) {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
          case POCL_FAKE_AS_PRIVATE:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            break;
          case POCL_FAKE_AS_GLOBAL:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
            break;
          case POCL_FAKE_AS_LOCAL:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
            break;
          case POCL_FAKE_AS_CONSTANT:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT;
            break;
          case POCL_FAKE_AS_GENERIC:
            current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            break;
#endif
          default:
            POCL_MSG_ERR("Unknown address space ID %lu\n", val);
            break;
          }
        }
      } else if (isa<MDString>(meta_node->getOperand(j))) {
        // std::cout << "is MDString" << std::endl;
        llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
        std::string val = m->getString().str();

        if (meta_name == "kernel_arg_access_qual") {
          assert(has_meta_for_every_arg &&
                 "kernel_arg_access_qual meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
          if (val == "read_write")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
          else if (val == "read_only")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_ONLY;
          else if (val == "write_only")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
          else if (val == "none")
            current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
          else
            std::cout << "UNKNOWN kernel_arg_access_qual value: " << val
                      << std::endl;
        } else if (meta_name == "kernel_arg_type") {
          assert(has_meta_for_every_arg && "kernel_arg_type meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
          current_arg->type_name = (char *)malloc(val.size() + 1);
          std::strcpy(current_arg->type_name, val.c_str());
        } else if (meta_name == "kernel_arg_base_type") {
          // may or may not be present even in SPIR
        } else if (meta_name == "kernel_arg_type_qual") {
          assert(has_meta_for_every_arg &&
                 "kernel_arg_type_qual meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
          current_arg->type_qualifier = 0;
          if (val.find("const") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
          if (val.find("restrict") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
          if (val.find("volatile") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
        } else if (meta_name == "kernel_arg_name") {
          assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
          kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
          current_arg->name = (char *)malloc(val.size() + 1);
          std::strcpy(current_arg->name, val.c_str());
        } else
          std::cout << "UNKNOWN opencl metadata name: " << meta_name
                    << std::endl;
      } else if (meta_name != "reqd_work_group_size")
        std::cout << "UNKNOWN opencl metadata class for: " << meta_name
                  << std::endl;
    }
  }
  return 0;
}

static std::map<std::string, unsigned> type_size_map = {
  {std::string("char"), (1)},
  {std::string("uchar"), (1)},
  {std::string("short"), (2)},
  {std::string("ushort"), (2)},
  {std::string("int"), (4)},
  {std::string("uint"), (4)},
  {std::string("long"), (8)},
  {std::string("ulong"), (8)},


  {std::string("char2"), (1*2)},
  {std::string("uchar2"), (1*2)},
  {std::string("short2"), (2*2)},
  {std::string("ushort2"), (2*2)},
  {std::string("int2"), (4*2)},
  {std::string("uint2"), (4*2)},
  {std::string("long2"), (8*2)},
  {std::string("ulong2"), (8*2)},


  {std::string("char3"), (1*4)},
  {std::string("uchar3"), (1*4)},
  {std::string("short3"), (2*4)},
  {std::string("ushort3"), (2*4)},
  {std::string("int3"), (4*4)},
  {std::string("uint3"), (4*4)},
  {std::string("long3"), (8*4)},
  {std::string("ulong3"), (8*4)},


  {std::string("char4"), (1*4)},
  {std::string("uchar4"), (1*4)},
  {std::string("short4"), (2*4)},
  {std::string("ushort4"), (2*4)},
  {std::string("int4"), (4*4)},
  {std::string("uint4"), (4*4)},
  {std::string("long4"), (8*4)},
  {std::string("ulong4"), (8*4)},


  {std::string("char8"), (1*8)},
  {std::string("uchar8"), (1*8)},
  {std::string("short8"), (2*8)},
  {std::string("ushort8"), (2*8)},
  {std::string("int8"), (4*8)},
  {std::string("uint8"), (4*8)},
  {std::string("long8"), (8*8)},
  {std::string("ulong8"), (8*8)},

  {std::string("char16"), (1*16)},
  {std::string("uchar16"), (1*16)},
  {std::string("short16"), (2*16)},
  {std::string("ushort16"), (2*16)},
  {std::string("int16"), (4*16)},
  {std::string("uint16"), (4*16)},
  {std::string("long16"), (8*16)},
  {std::string("ulong16"), (8*16)},

  {std::string("half"), (2)},
  {std::string("float"), (4)},
  {std::string("double"), (8)},

  {std::string("half2"), (2*2)},
  {std::string("float2"), (4*2)},
  {std::string("double2"), (8*2)},

  {std::string("half3"), (2*4)},
  {std::string("float3"), (4*4)},
  {std::string("double3"), (8*4)},

  {std::string("half4"), (2*4)},
  {std::string("float4"), (4*4)},
  {std::string("double4"), (8*4)},

  {std::string("half8"), (2*8)},
  {std::string("float8"), (4*8)},
  {std::string("double8"), (8*8)},

  {std::string("half16"), (2*16)},
  {std::string("float16"), (4*16)},
  {std::string("double16"), (8*16)}
};

#ifndef LLVM_OLDER_THAN_3_9
// Clang 3.9 uses function metadata instead of module metadata for presenting
// OpenCL kernel information.
static int pocl_get_kernel_arg_function_metadata(const char *kernel_name,
                                                 llvm::Module *input,
                                                 cl_kernel kernel) {
  llvm::Function *Kernel = NULL;

  // SPIR still uses the "opencl.kernels" MD.
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  int bitcode_is_old_spir = (opencl_kernels != NULL);

  if (bitcode_is_old_spir) {
    return pocl_get_kernel_arg_module_metadata(kernel_name, input, kernel);
  }
  // Else go on, because it might be SPIR encoded with modern LLVM

  for (llvm::Module::iterator i = input->begin(), e = input->end();
       i != e; ++i) {
    if (i->getMetadata("kernel_arg_access_qual") &&
        i->getName() == kernel_name) {
      Kernel = &*i;
      break;
    }
  }
  assert(Kernel);
  kernel->has_arg_metadata = 0;

  llvm::MDNode *meta_node;
  llvm::Value *meta_arg_value = NULL;
  struct pocl_argument_info *current_arg = NULL;

  kernel->arg_info = (struct pocl_argument_info *)calloc(
      kernel->num_args, sizeof(struct pocl_argument_info));
  memset(kernel->arg_info, 0,
         sizeof(struct pocl_argument_info) * kernel->num_args);

  // kernel_arg_addr_space
  meta_node = Kernel->getMetadata("kernel_arg_addr_space");
  assert(meta_node != nullptr);
  unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  int has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  for (unsigned j = 0; j < arg_num; ++j) {
    assert(has_meta_for_every_arg && "kernel_arg_addr_space meta incomplete");

    current_arg = &kernel->arg_info[j];
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
    // std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
    meta_arg_value =
        dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
    llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
    uint64_t val = m->getLimitedValue(UINT_MAX);

    bool SPIRAddressSpaceIDs;
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
    SPIRAddressSpaceIDs = bitcode_is_spir;
#else
    // We have an LLVM fixed to produce always SPIR AS ids for the argument
    // info metadata.
    SPIRAddressSpaceIDs = true;
#endif
    if (SPIRAddressSpaceIDs) {
      switch (val) {
      case 0:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        break;
      case 1:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
        break;
      case 3:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
        break;
      case 2:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT;
        break;
      default:
        POCL_MSG_ERR("Unknown address space ID %lu\n", val);
        break;
      }
    } else {
      switch (val) {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
      case POCL_FAKE_AS_PRIVATE:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        break;
      case POCL_FAKE_AS_GLOBAL:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
        break;
      case POCL_FAKE_AS_LOCAL:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
        break;
      case POCL_FAKE_AS_CONSTANT:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT;
        break;
      case POCL_FAKE_AS_GENERIC:
        current_arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        break;
#endif
      default:
        POCL_MSG_ERR("Unknown address space ID %lu\n", val);
        break;
      }
    }
  }

  // kernel_arg_access_qual
  meta_node = Kernel->getMetadata("kernel_arg_access_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");

  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    current_arg = &kernel->arg_info[j];
    // std::cout << "is MDString" << std::endl;
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
    if (val == "read_write")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
    else if (val == "read_only")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_ONLY;
    else if (val == "write_only")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
    else if (val == "none")
      current_arg->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
    else
      std::cout << "UNKNOWN kernel_arg_access_qual value: " << val << std::endl;
  }

  // kernel_arg_type
  meta_node = Kernel->getMetadata("kernel_arg_type");
  assert(meta_node != nullptr);
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type meta incomplete");

  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel->arg_info[j];
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
    current_arg->type_name = (char *)malloc(val.size() + 1);
    if (type_size_map.find(val) != type_size_map.end())
      current_arg->type_size = type_size_map[val];
    else
      current_arg->type_size = 0;
    std::strcpy(current_arg->type_name, val.c_str());
  }

  // kernel_arg_type_qual
  meta_node = Kernel->getMetadata("kernel_arg_type_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel->arg_info[j];
    assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
    kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
    current_arg->type_qualifier = 0;
    if (val.find("const") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
    if (val.find("restrict") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
    if (val.find("volatile") != std::string::npos)
      current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
  }

  // kernel_arg_name
  meta_node = Kernel->getMetadata("kernel_arg_name");
  if (meta_node) {
    arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
    has_meta_for_every_arg = (arg_num == kernel->num_args);
#endif
    assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
    for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
      llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
      std::string val = m->getString().str();

      current_arg = &kernel->arg_info[j];
      kernel->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
      current_arg->name = (char *)malloc(val.size() + 1);
      std::strcpy(current_arg->name, val.c_str());
    }
  }
  else {
    POCL_MSG_WARN("no name metadata for kernel args"
                  "(this is normal when compiling from SPIR) \n");
    // With SPIR 2.0 generated by more modern (3.9) Clang there is no more
    // "kernel_arg_name" metadata, so retrieve the name in another way
    // \todo Implement walking on the arguments themselves to get the names
  }

  return 0;
}
#endif

int pocl_llvm_get_kernel_metadata(cl_program program,
                                  cl_kernel kernel,
                                  int device_i,
                                  const char* kernel_name,
                                  int * errcode)
{
  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  int i;
  llvm::Module *input = NULL;
  cl_device_id Device = program->devices[device_i];

  assert(Device->llvm_target_triplet && "Device has no target triple set");

  if (program->llvm_irs != NULL && program->llvm_irs[device_i] != NULL)
    input = (llvm::Module *)program->llvm_irs[device_i];
  else {
    *errcode = CL_INVALID_PROGRAM_EXECUTABLE;
    return 1;
  }

  llvm::Function *KernelFunction = input->getFunction(kernel_name);
  if (!KernelFunction) {
    *errcode = CL_INVALID_KERNEL_NAME;
    return 1;
  }
  kernel->num_args = KernelFunction->arg_size();

#if defined(LLVM_OLDER_THAN_3_9)
  if (pocl_get_kernel_arg_module_metadata(kernel_name, input, kernel)) {
    *errcode = CL_INVALID_KERNEL;
    return 1;
  }
#else
  if (pocl_get_kernel_arg_function_metadata(kernel_name, input, kernel)) {
    *errcode = CL_INVALID_KERNEL;
    return 1;
  }
#endif

#ifdef DEBUG_POCL_LLVM_API
  printf("### fetching kernel metadata for kernel %s program %p "
         "input llvm::Module %p\n",
         kernel_name, program, input);
#endif

  DataLayout *TD = nullptr;
#ifdef LLVM_OLDER_THAN_3_7
  const std::string &ModuleDataLayout =
      input->getDataLayout()->getStringRepresentation();
#else
  const std::string &ModuleDataLayout =
      input->getDataLayout().getStringRepresentation();
#endif
  assert(!ModuleDataLayout.empty());
  TD = new DataLayout(ModuleDataLayout);

  SmallVector<GlobalVariable *, 8> locals;
  for (llvm::Module::global_iterator i = input->global_begin(),
                                     e = input->global_end();
       i != e; ++i) {
    std::string funcName = "";
    funcName = KernelFunction->getName().str();
    if (pocl::isAutomaticLocal(funcName, *i)) {
      POCL_MSG_PRINT_LLVM("Automatic local detected: %s\n",
                          i->getName().str().c_str());
      locals.push_back(&*i);
    }
  }

  kernel->num_locals = locals.size();

  /* Temporary store for the arguments that are set with clSetKernelArg. */
  kernel->dyn_arguments = (struct pocl_argument *)malloc(
      (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument));
  /* Initialize kernel "dynamic" arguments (in case the user doesn't). */
  for (unsigned i = 0; i < kernel->num_args; ++i) {
    kernel->dyn_arguments[i].value = NULL;
    kernel->dyn_arguments[i].size = 0;
  }

  /* Fill up automatic local arguments. */
  for (unsigned i = 0; i < kernel->num_locals; ++i) {
    unsigned auto_local_size =
        TD->getTypeAllocSize(locals[i]->getInitializer()->getType());
    kernel->dyn_arguments[kernel->num_args + i].value = NULL;
    kernel->dyn_arguments[kernel->num_args + i].size = auto_local_size;
#ifdef DEBUG_POCL_LLVM_API
    printf("### automatic local %d size %u\n", i, auto_local_size);
#endif
  }

  i = 0;
  for (llvm::Function::const_arg_iterator ii = KernelFunction->arg_begin(),
                                          ee = KernelFunction->arg_end();
       ii != ee; ii++) {
    llvm::Type *t = ii->getType();
    struct pocl_argument_info &ArgInfo = kernel->arg_info[i];
    ArgInfo.type = POCL_ARG_TYPE_NONE;
    ArgInfo.is_local = false;
    const llvm::PointerType *p = dyn_cast<llvm::PointerType>(t);
    if (p && !ii->hasByValAttr()) {
      ArgInfo.type = POCL_ARG_TYPE_POINTER;
      // index 0 is for function attributes, parameters start at 1.
      // TODO: detect the address space from MD.

#ifndef POCL_USE_FAKE_ADDR_SPACE_IDS
      if (ArgInfo.address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
        ArgInfo.is_local = true;
#else
      if (p->getAddressSpace() == POCL_FAKE_AS_GLOBAL ||
          p->getAddressSpace() == POCL_FAKE_AS_CONSTANT ||
          pocl::is_image_type(*t) || pocl::is_sampler_type(*t)) {
        kernel->arg_info[i].is_local = false;
      } else {
        if (p->getAddressSpace() != POCL_FAKE_AS_LOCAL) {
          p->dump();
          assert(p->getAddressSpace() == POCL_FAKE_AS_LOCAL);
        }
        kernel->arg_info[i].is_local = true;
      }
#endif
    }

    if (pocl::is_image_type(*t)) {
      ArgInfo.type = POCL_ARG_TYPE_IMAGE;
    } else if (pocl::is_sampler_type(*t)) {
      ArgInfo.type = POCL_ARG_TYPE_SAMPLER;
    }
    i++;
  }

  std::stringstream attrstr;
  std::string vectypehint;
  // fill 'kernel->reqd_wg_size'
  kernel->reqd_wg_size = (size_t *)malloc(3 * sizeof(size_t));

  size_t reqdx = 0, reqdy = 0, reqdz = 0;

#ifdef LLVM_OLDER_THAN_3_9
  llvm::NamedMDNode *size_info =
    KernelFunction->getParent()->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      llvm::MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (dyn_cast<ValueAsMetadata>(
        KernelSizeInfo->getOperand(0).get())->getValue() != KernelFunction)
        continue;
      reqdx = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(1))->getValue()))->getLimitedValue();
      reqdy = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(2))->getValue()))->getLimitedValue();
      reqdz = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   KernelSizeInfo->getOperand(3))->getValue()))->getLimitedValue();
      break;
    }
  }
#else
  llvm::MDNode *ReqdWGSize =
      KernelFunction->getMetadata("reqd_work_group_size");
  if (ReqdWGSize != NULL) {
    reqdx = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(0))->getValue()))->getLimitedValue();
    reqdy = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(1))->getValue()))->getLimitedValue();
    reqdz = (llvm::cast<ConstantInt>(
               llvm::dyn_cast<ConstantAsMetadata>(
                 ReqdWGSize->getOperand(2))->getValue()))->getLimitedValue();
  }
#endif

  // TODO: implement vec_type_hint / work_group_size_hint attributes
  kernel->reqd_wg_size[0] = reqdx;
  kernel->reqd_wg_size[1] = reqdy;
  kernel->reqd_wg_size[2] = reqdz;
  if (reqdx || reqdy || reqdz)
    attrstr << "__attribute__((reqd_work_group_size("
            << reqdx << ", " << reqdy
            << ", " << reqdz << " )))";
  if (vectypehint.size() > 0) {
    if (reqdx || reqdy || reqdz)
      attrstr << " ";
    attrstr << "__attribute__ ((vec_type_hint (" << vectypehint << ")))";
  }

  std::string r = attrstr.str();
  if (r.size() > 0) {
    kernel->attributes = (char *)malloc(r.size() + 1);
    std::memcpy(kernel->attributes, r.c_str(), r.size());
    kernel->attributes[r.size()] = 0;
  } else
    kernel->attributes = NULL;

#ifndef POCL_ANDROID
  // Generate the kernel_obj.c file. This should be optional
  // and generated only for the heterogeneous standalone devices which
  // need the definitions to accompany the kernels, for the launcher
  // code.
  // TODO: the scripts use a generated kernel.h header file that
  // gets added to this file. No checks seem to fail if that file
  // is missing though, so it is left out from there for now

  std::stringstream content;

  content << std::endl
          << "#include <pocl_device.h>" << std::endl
          << "void _pocl_launcher_" << kernel_name
          << "_workgroup(void** args, struct pocl_context*);" << std::endl
          << "void _pocl_launcher_" << kernel_name
          << "_workgroup_fast(void** args, struct pocl_context*);" << std::endl;

  if (Device->global_as_id != 0)
    content << "__attribute__((address_space(" << Device->global_as_id << ")))"
            << std::endl;

  content << "__kernel_metadata _" << kernel_name << "_md = {" << std::endl
          << "     \"" << kernel_name << "\"," << std::endl
          << "     " << kernel->num_args << "," << std::endl
          << "     " << kernel->num_locals << "," << std::endl
          << "     _pocl_launcher_" << kernel_name << "_workgroup_fast"
          << std::endl
          << " };" << std::endl;

  pocl_cache_write_descriptor(program, device_i, kernel_name,
                              content.str().c_str(), content.str().size());
#endif

  delete TD;
  *errcode = CL_SUCCESS;
  return 0;
}



/* This is the implementation of the public pocl_llvm_get_kernel_count(),
 * and is used internally also by pocl_llvm_get_kernel_names to
 */
static unsigned pocl_llvm_get_kernel_count(cl_program program,
                                           char **knames,
                                           unsigned max_num_krn)
{
  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  /* any device's module will do for metadata, just use first non-NULL */
  llvm::Module *mod = NULL;
  unsigned i;
  for (i = 0; i < program->num_devices; i++)
    if (program->llvm_irs[i]) {
      mod = (llvm::Module *)program->llvm_irs[i];
      break;
    }

  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");
  if (md) {

    if (knames) {
      for (unsigned i = 0; i < max_num_krn; i++) {
        assert(md->getOperand(i)->getOperand(0) != NULL);
        llvm::ValueAsMetadata *value =
            dyn_cast<llvm::ValueAsMetadata>(md->getOperand(i)->getOperand(0));
        llvm::Function *k = cast<Function>(value->getValue());
        knames[i] = strdup(k->getName().data());
      }
    }
    return md->getNumOperands();
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    unsigned kernel_count = 0;
    for (llvm::Module::iterator i = mod->begin(), e = mod->end(); i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")) {
        if (knames && kernel_count < max_num_krn) {
          knames[kernel_count] = strdup(i->getName().str().c_str());
        }
        ++kernel_count;
      }
    }
    return kernel_count;
  }
}

unsigned pocl_llvm_get_kernel_count(cl_program program) {
  return pocl_llvm_get_kernel_count(program, NULL, 0);
}

unsigned pocl_llvm_get_kernel_names(cl_program program, char **knames,
                                    unsigned max_num_krn) {
  unsigned n = pocl_llvm_get_kernel_count(program, knames, max_num_krn);

  return n;
}
