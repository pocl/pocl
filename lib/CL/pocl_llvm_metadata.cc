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

#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "pocl_cl.h"
#include "pocl_llvm_api.h"
#include "pocl_cache.h"
#include "LLVMUtils.h"

using namespace llvm;

static inline bool is_image_type(llvm::Type *ArgType,
                                 struct pocl_argument_info &ArgInfo,
                                 cl_bitfield has_arg_meta) {
  if (ArgType->isPointerTy() || ArgType->isTargetExtTy()) {
    assert(has_arg_meta & POCL_HAS_KERNEL_ARG_TYPE_NAME);
    llvm::StringRef name(ArgInfo.type_name);
    if ((has_arg_meta & POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER) &&
        (ArgInfo.access_qualifier != CL_KERNEL_ARG_ACCESS_NONE)) {
      if (name.starts_with("image2d_") || name.starts_with("image3d_") ||
          name.starts_with("image1d_") || name.starts_with("_pocl_image"))
        return true;
    }
  }
  return false;
}

static inline bool is_sampler_type(struct pocl_argument_info &ArgInfo,
                                   cl_bitfield has_arg_meta) {
  assert(has_arg_meta & POCL_HAS_KERNEL_ARG_TYPE_NAME);
  llvm::StringRef name(ArgInfo.type_name);
  return name == "sampler_t";
}

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

    bool has_name_metadata = true;
    if ((kernel_meta->has_arg_metadata & POCL_HAS_KERNEL_ARG_NAME) == 0) {
      for (unsigned j = 0; j < arg_num; ++j) {
        struct pocl_argument_info *current_arg = &kernel_meta->arg_info[j];
        Argument* Arg = Kernel->getArg(j);
        if (Arg->hasName()) {
          const char *ArgName = Arg->getName().data();
          current_arg->name = strdup(ArgName);
        } else {
          has_name_metadata = false;
          break;
        }
      }
      if (has_name_metadata)
        kernel_meta->has_arg_metadata |= POCL_HAS_KERNEL_ARG_NAME;
    }

  }
  return 0;
}

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
    POCL_MSG_PRINT_LLVM("no name metadata for kernel args"
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

  if (program->llvm_irs[device_i] != nullptr)
    input = static_cast<llvm::Module *>(program->llvm_irs[device_i]);
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
      if (kernel->getName() != POCL_GVAR_INIT_KERNEL_NAME)
        kernels.push_back(kernel);
    }
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    for (llvm::Module::iterator i = input->begin(), e = input->end(); i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")
          && i->getName() != POCL_GVAR_INIT_KERNEL_NAME) {
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
    meta->local_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));
    meta->private_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));
    meta->spill_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));

    llvm::Function *KernelFunction = kernels[j];

    meta->num_args = KernelFunction->arg_size();
    std::string funcName = KernelFunction->getName().str();
    meta->name = strdup(funcName.c_str());

    if (pocl_get_kernel_arg_function_metadata(KernelFunction, input, meta)) {
      return CL_INVALID_KERNEL;
    }

#ifdef DEBUG_POCL_LLVM_API
    printf("### fetching kernel metadata for kernel %s program %p "
           "input llvm::Module %p\n",
           kernel_name, program, input);
#endif

    locals.clear();

    for (llvm::Module::global_iterator i = input->global_begin(),
                                       e = input->global_end();
         i != e; ++i) {
      if (pocl::isAutomaticLocal(KernelFunction, *i)) {
        POCL_MSG_PRINT_LLVM("Automatic local detected in kernel %s: %s\n",
                            meta->name, (*i).getName().data());
        locals.push_back(&*i);
      }
    }

    meta->num_locals = locals.size();
    meta->local_sizes = (size_t*)calloc(locals.size(), sizeof(size_t));
    size_t total_local_size = 0;

    /* Fill up automatic local arguments. */
    for (unsigned i = 0; i < meta->num_locals; ++i) {
      unsigned auto_local_size =
          TD->getTypeAllocSize(locals[i]->getInitializer()->getType());
      meta->local_sizes[i] = auto_local_size;
      total_local_size += auto_local_size;
#ifdef DEBUG_POCL_LLVM_API
      printf("### automatic local %d size %u\n", i, auto_local_size);
#endif
    }
    meta->local_mem_size[device_i] = total_local_size;

    i = 0;
    for (llvm::Function::const_arg_iterator ii = KernelFunction->arg_begin(),
                                            ee = KernelFunction->arg_end();
         ii != ee; ii++) {
      llvm::Type *ARGt = ii->getType();
      if (ii->hasByValAttr())
        ARGt = ii->getParamByValType();
      if (ii->hasByRefAttr())
        ARGt = ii->getParamByRefType();
      if (ii->hasInAllocaAttr())
        ARGt = ii->getParamInAllocaType();
      if (ii->hasStructRetAttr())
        ARGt = ii->getParamStructRetType();

      struct pocl_argument_info &ArgInfo = meta->arg_info[i];
      ArgInfo.type = POCL_ARG_TYPE_NONE;
      ArgInfo.type_size = 0;
      const llvm::PointerType *ARGp = dyn_cast<llvm::PointerType>(ARGt);

      if (is_image_type(ARGt, ArgInfo, meta->has_arg_metadata)) {
        ArgInfo.type = POCL_ARG_TYPE_IMAGE;
        ArgInfo.type_size = sizeof(cl_mem);
      } else
      if (is_sampler_type(ArgInfo, meta->has_arg_metadata)) {
        ArgInfo.type = POCL_ARG_TYPE_SAMPLER;
        ArgInfo.type_size = sizeof(cl_sampler);
      } else

      if (ARGp) {
        ArgInfo.type = POCL_ARG_TYPE_POINTER;
        ArgInfo.type_size = sizeof(cl_mem);
      // structs, classes and arrays are missing; calculating
      // their size is not trivial
      } else if (ARGt->isSized() && ARGt->isSingleValueType()) {
        TypeSize TS = input->getDataLayout().getTypeAllocSize(ARGt);
        ArgInfo.type_size = TS.getFixedValue();
      } else {
        POCL_MSG_PRINT_LLVM(
            "Arg %u (%s) : Don't know how to determine type size\n", i,
            ArgInfo.type_name);
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

    if (vectypehint.size() > 0) {
      strncpy(meta->vectypehint, vectypehint.c_str(),
              sizeof(meta->vectypehint));
      meta->vectypehint[sizeof(meta->vectypehint) - 1] = 0;
      if (attrstr.tellp() > 0)
        attrstr << " ";
      attrstr << "__attribute__ ((vec_type_hint (" << vectypehint << ")))";
    }

    std::string r = attrstr.str();
    if (r.size() > 0) {
      meta->attributes = (char *)malloc(r.size() + 1);
      std::memcpy(meta->attributes, r.c_str(), r.size());
      meta->attributes[r.size()] = 0;
    } else
      meta->attributes = nullptr;

    /* If the program is compiled with -cl-opt-disable, or the opt
     * has some problem that hinders optimization, allocas might
     * not be optimized away at all. In that case, the estimated
     * stack size might be the actual stack size.
     * Set the kernel limit on workgroup size accordingly. */
    if (Device->work_group_stack_size > 0) {
      unsigned long EstStackSize = 0;
      std::string MetadataKey(meta->name);
      MetadataKey.append(".meta.est.stack.size");
      if (getModuleIntMetadata(*input, MetadataKey.c_str(), EstStackSize) &&
          EstStackSize > 0) {
        size_t WorkItemsStackLimit =
            Device->work_group_stack_size / EstStackSize;
        if (meta->max_workgroup_size == nullptr)
          meta->max_workgroup_size = (size_t *)malloc(
              sizeof(size_t) * program->associated_num_devices);
        // The estimate is the worst-case only; keep the minimum of 1 work item
        meta->max_workgroup_size[device_i] =
            std::max((size_t)1, std::min((size_t)Device->max_work_group_size,
                                         WorkItemsStackLimit));
        POCL_MSG_PRINT_LLVM("Kernel %s: limited max WG size to %zu \n",
                            meta->name, meta->max_workgroup_size[device_i]);
      }
    }
  } // for each kernel

  delete TD;

  return CL_SUCCESS;
}


unsigned pocl_llvm_get_kernel_count(cl_program program, unsigned device_i) {

  cl_context ctx = program->context;
  PoclLLVMContextData *llvm_ctx = (PoclLLVMContextData *)ctx->llvm_context_data;
  PoclCompilerMutexGuard lockHolder(&llvm_ctx->Lock);

  /* any device's module will do for metadata, just use first non-nullptr */
  llvm::Module *mod = nullptr;
  if (program->llvm_irs[device_i] != nullptr)
    mod = static_cast<llvm::Module *>(program->llvm_irs[device_i]);
  else {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }

  llvm::NamedMDNode *md = mod->getNamedMetadata("opencl.kernels");
  if (md) {
    return md->getNumOperands();
  }
  // LLVM 3.9 does not use opencl.kernels meta, but kernel_arg_* function meta
  else {
    unsigned kernel_count = 0;
    for (llvm::Module::iterator i = mod->begin(), e = mod->end(); i != e; ++i) {
      if (i->getMetadata("kernel_arg_access_qual")
          && i->getName() != POCL_GVAR_INIT_KERNEL_NAME) {
        ++kernel_count;
      }
    }
    return kernel_count;
  }
}
