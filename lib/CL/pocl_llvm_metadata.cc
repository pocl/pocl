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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/Support/Casting.h>
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
static int pocl_get_kernel_arg_module_metadata(llvm::Function *Kernel,
                                               llvm::Module *input,
                                               pocl_kernel_metadata_t *kernel_meta) {
  // find the right kernel in "opencl.kernels" metadata
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  llvm::MDNode *kernel_metadata = nullptr;

  if (!(opencl_kernels && opencl_kernels->getNumOperands()))
    // Perhaps it is a SPIR kernel without the "opencl.kernels" metadata
    return 1;

  for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
    llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

    llvm::Value *meta =
        dyn_cast<llvm::ValueAsMetadata>(kernel_iter->getOperand(0))->getValue();
    llvm::Function *temp = llvm::cast<llvm::Function>(meta);
    if (temp->getName().compare(Kernel->getName()) == 0) {
      kernel_metadata = kernel_iter;
      break;
    }
  }

  kernel_meta->arg_info = (struct pocl_argument_info *)calloc(
      kernel_meta->num_args, sizeof(struct pocl_argument_info));
  memset(kernel_meta->arg_info, 0,
         sizeof(struct pocl_argument_info) * kernel_meta->num_args);

  kernel_meta->has_arg_metadata = 0;

  assert(kernel_metadata && "kernel NOT found in opencl.kernels metadata");

  unsigned e = kernel_metadata->getNumOperands();
  for (unsigned i = 1; i != e; ++i) {
    llvm::MDNode *meta_node =
        llvm::cast<MDNode>(kernel_metadata->getOperand(i));

    // argument num
    unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
    int has_meta_for_every_arg = ((arg_num - 1) == kernel_meta->num_args);
#endif

    llvm::MDString *meta_name_node =
        llvm::cast<MDString>(meta_node->getOperand(0));
    std::string meta_name = meta_name_node->getString().str();

    for (unsigned j = 1; j != arg_num; ++j) {
      llvm::Value *meta_arg_value = nullptr;
      if (isa<ValueAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
            dyn_cast<ValueAsMetadata>(meta_node->getOperand(j))->getValue();
      else if (isa<ConstantAsMetadata>(meta_node->getOperand(j)))
        meta_arg_value =
            dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
      struct pocl_argument_info *current_arg = &kernel_meta->arg_info[j - 1];

      if (meta_arg_value != nullptr && isa<ConstantInt>(meta_arg_value) &&
          meta_name == "kernel_arg_addr_space") {
        assert(has_meta_for_every_arg &&
               "kernel_arg_addr_space meta incomplete");
        kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
        // std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
        llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
        unsigned long val = (unsigned long)m->getLimitedValue(UINT_MAX);
        // We have an LLVM fixed to produce always SPIR AS ids for the argument
        // info metadata.
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

      } else if (isa<MDString>(meta_node->getOperand(j))) {
        // std::cout << "is MDString" << std::endl;
        llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
        std::string val = m->getString().str();

        if (meta_name == "kernel_arg_access_qual") {
          assert(has_meta_for_every_arg &&
                 "kernel_arg_access_qual meta incomplete");
          kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
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
          kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
          current_arg->type_name = (char *)malloc(val.size() + 1);
          std::strcpy(current_arg->type_name, val.c_str());
        } else if (meta_name == "kernel_arg_base_type") {
          // may or may not be present even in SPIR
        } else if (meta_name == "kernel_arg_type_qual") {
          assert(has_meta_for_every_arg &&
                 "kernel_arg_type_qual meta incomplete");
          kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
          current_arg->type_qualifier = 0;
          if (val.find("const") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
          if (val.find("restrict") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
          if (val.find("volatile") != std::string::npos)
            current_arg->type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
        } else if (meta_name == "kernel_arg_name") {
          assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
          kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
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

// Clang 3.9 uses function metadata instead of module metadata for presenting
// OpenCL kernel information.
static int pocl_get_kernel_arg_function_metadata(llvm::Function *Kernel,
                                                 llvm::Module *input,
                                                 pocl_kernel_metadata_t *kernel_meta) {
  assert(Kernel);

  // SPIR still uses the "opencl.kernels" MD.
  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  int bitcode_is_old_spir = (opencl_kernels != nullptr);

  if (bitcode_is_old_spir) {
    return pocl_get_kernel_arg_module_metadata(Kernel, input, kernel_meta);
  }
  // Else go on, because it might be SPIR encoded with modern LLVM

  kernel_meta->has_arg_metadata = 0;

  llvm::MDNode *meta_node;
  llvm::Value *meta_arg_value = nullptr;
  struct pocl_argument_info *current_arg = nullptr;

  kernel_meta->arg_info = (struct pocl_argument_info *)calloc(
      kernel_meta->num_args, sizeof(struct pocl_argument_info));
  memset(kernel_meta->arg_info, 0,
         sizeof(struct pocl_argument_info) * kernel_meta->num_args);

  // kernel_arg_addr_space
  meta_node = Kernel->getMetadata("kernel_arg_addr_space");
  assert(meta_node != nullptr);
  unsigned arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  int has_meta_for_every_arg = (arg_num == kernel_meta->num_args);
#endif
  for (unsigned j = 0; j < arg_num; ++j) {
    assert(has_meta_for_every_arg && "kernel_arg_addr_space meta incomplete");

    current_arg = &kernel_meta->arg_info[j];
    kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER;
    // std::cout << "is ConstantInt /  kernel_arg_addr_space" << std::endl;
    meta_arg_value =
        dyn_cast<ConstantAsMetadata>(meta_node->getOperand(j))->getValue();
    llvm::ConstantInt *m = llvm::cast<ConstantInt>(meta_arg_value);
    unsigned long val = (unsigned long)m->getLimitedValue(UINT_MAX);

    // We have an LLVM fixed to produce always SPIR AS ids for the argument
    // info metadata.
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
  }

  // kernel_arg_access_qual
  meta_node = Kernel->getMetadata("kernel_arg_access_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel_meta->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");

  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    current_arg = &kernel_meta->arg_info[j];
    // std::cout << "is MDString" << std::endl;
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    assert(has_meta_for_every_arg && "kernel_arg_access_qual meta incomplete");
    kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
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
  has_meta_for_every_arg = (arg_num == kernel_meta->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type meta incomplete");

  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel_meta->arg_info[j];
    kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_NAME;
    current_arg->type_name = (char *)malloc(val.size() + 1);
    if (current_arg->address_qualifier != CL_KERNEL_ARG_ADDRESS_PRIVATE) {
      current_arg->type_size = sizeof(void *);
    } else if (type_size_map.find(val) != type_size_map.end()) {
      current_arg->type_size = type_size_map[val];
    } else {
      current_arg->type_size = 0;
    }
    std::strcpy(current_arg->type_name, val.c_str());
  }

  // kernel_arg_type_qual
  meta_node = Kernel->getMetadata("kernel_arg_type_qual");
  arg_num = meta_node->getNumOperands();
#ifndef NDEBUG
  has_meta_for_every_arg = (arg_num == kernel_meta->num_args);
#endif
  assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
  for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
    llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
    std::string val = m->getString().str();

    current_arg = &kernel_meta->arg_info[j];
    assert(has_meta_for_every_arg && "kernel_arg_type_qual meta incomplete");
    kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER;
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
    has_meta_for_every_arg = (arg_num == kernel_meta->num_args);
#endif
    assert(has_meta_for_every_arg && "kernel_arg_name meta incomplete");
    for (unsigned j = 0; j < meta_node->getNumOperands(); ++j) {
      llvm::MDString *m = llvm::cast<MDString>(meta_node->getOperand(j));
      std::string val = m->getString().str();

      current_arg = &kernel_meta->arg_info[j];
      kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
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

/*****************************************************************************/

int pocl_llvm_get_kernels_metadata(cl_program program, unsigned device_i) {

  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  unsigned i,j;
  llvm::Module *input = nullptr;

  cl_device_id Device = program->devices[device_i];
  assert(Device->llvm_target_triplet && "Device has no target triple set");

  if (program->data != nullptr && program->data[device_i] != nullptr)
    input = static_cast<llvm::Module *>(program->data[device_i]);
  else {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }

  DataLayout *TD = nullptr;
  const std::string &ModuleDataLayout =
      input->getDataLayout().getStringRepresentation();
  assert(!ModuleDataLayout.empty());
  TD = new DataLayout(ModuleDataLayout);

  std::vector<llvm::Function *> kernels;

  llvm::NamedMDNode *opencl_kernels = input->getNamedMetadata("opencl.kernels");
  if (opencl_kernels) {
    for (unsigned i = 0, e = opencl_kernels->getNumOperands(); i != e; ++i) {
      llvm::MDNode *kernel_iter = opencl_kernels->getOperand(i);

      llvm::Value *meta =
          dyn_cast<llvm::ValueAsMetadata>(kernel_iter->getOperand(0))->getValue();
      llvm::Function *kernel = llvm::cast<llvm::Function>(meta);
      //kernel_names.push_back(kernel_prototype->getName().str());
      kernels.push_back(kernel);
    }
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    for (llvm::Module::iterator i = input->begin(), e = input->end(); i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")) {
         kernels.push_back(&*i);
      }
    }
  }

  assert ((kernels.size() > 0) && "empty program - no kernels");
  assert (kernels.size() == program->num_kernels);

/**************************************************************************/

  SmallVector<GlobalVariable *, 8> locals;

  for (j = 0; j < program->num_kernels; ++j) {

    pocl_kernel_metadata_t *meta = &program->kernel_meta[j];
    meta->data = (void**)calloc(program->num_devices, sizeof(void*));

    llvm::Function *KernelFunction = kernels[j];

    meta->num_args = KernelFunction->arg_size();
    meta->name = strdup(KernelFunction->getName().str().c_str());

    if (pocl_get_kernel_arg_function_metadata(KernelFunction, input, meta)) {
      return CL_INVALID_KERNEL;
    }

#ifdef DEBUG_POCL_LLVM_API
    printf("### fetching kernel metadata for kernel %s program %p "
           "input llvm::Module %p\n",
           kernel_name, program, input);
#endif

    locals.clear();

    std::string funcName = KernelFunction->getName().str();

    for (llvm::Module::global_iterator i = input->global_begin(),
                                       e = input->global_end();
         i != e; ++i) {
      if (pocl::isAutomaticLocal(funcName, *i)) {
        POCL_MSG_PRINT_LLVM("Automatic local detected in kernel %s: %s\n",
                            meta->name, (*i).getName().data());
        locals.push_back(&*i);
      }
    }

    meta->num_locals = locals.size();
    meta->local_sizes = (size_t*)calloc(locals.size(), sizeof(size_t));

    /* Fill up automatic local arguments. */
    for (unsigned i = 0; i < meta->num_locals; ++i) {
      unsigned auto_local_size =
          TD->getTypeAllocSize(locals[i]->getInitializer()->getType());
      meta->local_sizes[i] = auto_local_size;

      #ifdef DEBUG_POCL_LLVM_API
          printf("### automatic local %d size %u\n", i, auto_local_size);
      #endif
    }

    i = 0;
    for (llvm::Function::const_arg_iterator ii = KernelFunction->arg_begin(),
                                            ee = KernelFunction->arg_end();
         ii != ee; ii++) {
      llvm::Type *t = ii->getType();
      struct pocl_argument_info &ArgInfo = meta->arg_info[i];
      ArgInfo.type = POCL_ARG_TYPE_NONE;
      const llvm::PointerType *p = dyn_cast<llvm::PointerType>(t);
      if (p && !ii->hasByValAttr()) {
        ArgInfo.type = POCL_ARG_TYPE_POINTER;
        // index 0 is for function attributes, parameters start at 1.
        // TODO: detect the address space from MD.
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
    std::string wgsizehint;

    size_t reqdx = 0, reqdy = 0, reqdz = 0;
    size_t wghintx = 0, wghinty = 0, wghintz = 0;

    llvm::MDNode *ReqdWGSize =
        KernelFunction->getMetadata("reqd_work_group_size");
    if (ReqdWGSize != nullptr) {
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

    llvm::MDNode *WGSizeHint =
        KernelFunction->getMetadata("work_group_size_hint");
    if (WGSizeHint != nullptr) {
      wghintx = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   WGSizeHint->getOperand(0))->getValue()))->getLimitedValue();
      wghinty = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   WGSizeHint->getOperand(1))->getValue()))->getLimitedValue();
      wghintz = (llvm::cast<ConstantInt>(
                 llvm::dyn_cast<ConstantAsMetadata>(
                   WGSizeHint->getOperand(2))->getValue()))->getLimitedValue();
    }

#ifndef LLVM_OLDER_THAN_11_0
    llvm::MDNode *VecTypeHint = KernelFunction->getMetadata("vec_type_hint");
    if (VecTypeHint != nullptr) {
      llvm::Value *VTHvalue = nullptr;
      if (isa<ValueAsMetadata>(VecTypeHint->getOperand(0))) {
        llvm::Value *val =
            dyn_cast<ValueAsMetadata>(VecTypeHint->getOperand(0))->getValue();
        llvm::Type *ty = val->getType();
        llvm::FixedVectorType *VectorTy = nullptr;
        if (ty != nullptr)
          VectorTy = dyn_cast<llvm::FixedVectorType>(ty);

        if (VectorTy) {
          llvm::Type *ElemType = VectorTy->getElementType();
          switch (ElemType->getTypeID()) {
          case Type::TypeID::HalfTyID:
            vectypehint = "half";
            break;
          case Type::TypeID::FloatTyID:
            vectypehint = "float";
            break;
          case Type::TypeID::DoubleTyID:
            vectypehint = "double";
            break;
          case Type::TypeID::IntegerTyID: {
            const llvm::MDOperand &SignMD = VecTypeHint->getOperand(1);
            llvm::Constant *SignVal =
              dyn_cast<ConstantAsMetadata>(SignMD)->getValue();
            llvm::ConstantInt *SignValInt = llvm::cast<ConstantInt>(SignVal);
            assert(SignValInt);
            if (SignValInt->getLimitedValue() == 0)
              vectypehint = "u";
            else
              vectypehint.clear();
            switch (ElemType->getIntegerBitWidth()) {
            case 8:
              vectypehint += "char";
              break;
            case 16:
              vectypehint += "short";
              break;
            case 32:
              vectypehint += "int";
              break;
            case 64:
              vectypehint += "long";
              break;
            default:
              vectypehint += "unknownInt";
              break;
            }
            break;
          }
          default:
            vectypehint = "unknownType";
            break;
          }
          switch (VectorTy->getNumElements()) {
          case 1:
            break;
          case 2:
            vectypehint += "2";
            break;
          case 3:
            vectypehint += "3";
            break;
          case 4:
            vectypehint += "4";
            break;
          case 8:
            vectypehint += "8";
            break;
          case 16:
            vectypehint += "16";
            break;
          default:
            vectypehint += "99";
            break;
          }
        }
      }
    }
#endif

    if (reqdx || reqdy || reqdz) {
      meta->reqd_wg_size[0] = reqdx;
      meta->reqd_wg_size[1] = reqdy;
      meta->reqd_wg_size[2] = reqdz;
      attrstr << "__attribute__((reqd_work_group_size("
              << reqdx << ", " << reqdy
              << ", " << reqdz << " )))";
    }

    if (wghintx || wghinty || wghintz) {
      meta->wg_size_hint[0] = wghintx;
      meta->wg_size_hint[1] = wghinty;
      meta->wg_size_hint[2] = wghintz;
      if (attrstr.tellp() > 0)
        attrstr << " ";
      attrstr << "__attribute__((work_group_size_hint("
              << wghintx << ", " << wghinty
              << ", " << wghintz << " )))";
    }

#ifndef LLVM_OLDER_THAN_11_0
    if (vectypehint.size() > 0) {
      strncpy(meta->vectypehint, vectypehint.c_str(),
              sizeof(meta->vectypehint));
      if (attrstr.tellp() > 0)
        attrstr << " ";
      attrstr << "__attribute__ ((vec_type_hint (" << vectypehint << ")))";
    }
#endif

    std::string r = attrstr.str();
    if (r.size() > 0) {
      meta->attributes = (char *)malloc(r.size() + 1);
      std::memcpy(meta->attributes, r.c_str(), r.size());
      meta->attributes[r.size()] = 0;
    } else
      meta->attributes = nullptr;

  } // for each kernel

  delete TD;

  return CL_SUCCESS;
}


unsigned pocl_llvm_get_kernel_count(cl_program program, unsigned device_i) {

  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  /* any device's module will do for metadata, just use first non-nullptr */
  llvm::Module *mod = (llvm::Module *)program->data[device_i];
  if (mod == nullptr)
    return 0;

  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");
  if (md) {
    return md->getNumOperands();
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    unsigned kernel_count = 0;
    for (llvm::Module::iterator i = mod->begin(), e = mod->end(); i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")) {
        ++kernel_count;
      }
    }
    return kernel_count;
  }
}
