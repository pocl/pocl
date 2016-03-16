/* pocl-ptx-gen.c - PTX code generation functions

   Copyright (c) 2016 James Price / University of Bristol

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

#include "config.h"

#include "common.h"
#include "pocl.h"
#include "pocl-ptx-gen.h"

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"

// TODO: Should these be proper passes?
void pocl_add_kernel_annotations(llvm::Module *module);
void pocl_fix_constant_address_space(llvm::Module *module);
void pocl_gen_local_mem_args(llvm::Module *module);
void pocl_insert_ptx_intrinsics(llvm::Module *module);

int pocl_ptx_gen(const char *bc_filename,
                 const char *ptx_filename,
                 const char *gpu_arch)
{
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
    llvm::MemoryBuffer::getFile(bc_filename);
  if (!buffer)
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to open bitcode file\n");
    return 1;
  }

  // Load bitcode
  llvm::ErrorOr<std::unique_ptr<llvm::Module>> module =
    parseBitcodeFile(buffer->get()->getMemBufferRef(),
    llvm::getGlobalContext());
  if (!module)
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return 1;
  }

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  // Apply transforms to prepare for lowering to PTX
  pocl_fix_constant_address_space(module->get());
  pocl_gen_local_mem_args(module->get());
  pocl_insert_ptx_intrinsics(module->get());
  pocl_add_kernel_annotations(module->get());
  //(*module)->dump();

  // TODO: support 32-bit?
  llvm::StringRef triple = "nvptx64-nvidia-cuda";

  // Get NVPTX target
  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
    triple, error);
  if (!target)
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to get target\n");
    POCL_MSG_ERR("%s\n", error.c_str());
    return 1;
  }

  // TODO: set options
  llvm::TargetOptions options;

  // TODO: CPU and features?
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, gpu_arch, "", options));

  llvm::legacy::PassManager passes;

  // Add pass to emit PTX
  std::error_code ec;
  llvm::raw_fd_ostream *ptx =
    new llvm::raw_fd_ostream(ptx_filename, ec, llvm::sys::fs::F_Text);
  if (machine->addPassesToEmitFile(passes, *ptx,
                                   llvm::TargetMachine::CGFT_AssemblyFile))
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to add passes\n");
    return 1;
  }

  // Run passes
  passes.run(**module);

  return 0;
}

void pocl_add_kernel_annotations(llvm::Module *module)
{
  llvm::LLVMContext& context = llvm::getGlobalContext();

  // Remove nvvm.annotations metadata since it is sometimes corrupt
  llvm::NamedMDNode *nvvm_annotations =
    module->getNamedMetadata("nvvm.annotations");
  if (nvvm_annotations)
    nvvm_annotations->eraseFromParent();

  llvm::NamedMDNode *md_kernels = module->getNamedMetadata("opencl.kernels");
  if (!md_kernels)
    return;

  // Add nvvm.annotations metadata to mark kernel entry points
  nvvm_annotations = module->getOrInsertNamedMetadata("nvvm.annotations");
  for (auto K = md_kernels->op_begin(); K != md_kernels->op_end(); K++)
  {
    if (!(*K)->getOperand(0))
      continue;

    llvm::ConstantAsMetadata *cam =
      llvm::dyn_cast<llvm::ConstantAsMetadata>((*K)->getOperand(0).get());
    if (!cam)
      continue;

    llvm::Function *function = llvm::dyn_cast<llvm::Function>(cam->getValue());

    llvm::Constant *one =
      llvm::ConstantInt::getSigned(llvm::Type::getInt32Ty(context), 1);
    llvm::Metadata *md_f = llvm::ValueAsMetadata::get(function);
    llvm::Metadata *md_n = llvm::MDString::get(context, "kernel");
    llvm::Metadata *md_1 = llvm::ConstantAsMetadata::get(one);

    std::vector<llvm::Metadata*> v_md = {md_f, md_n, md_1};
    llvm::MDNode *node = llvm::MDNode::get(context, v_md);
    nvvm_annotations->addOperand(node);
  }
}

void pocl_update_users_address_space(llvm::Value *inst)
{
  std::vector<llvm::Value*> users(inst->users().begin(), inst->users().end());
  for (auto U = users.begin(); U != users.end(); U++)
  {
    if (auto bitcast = llvm::dyn_cast<llvm::BitCastInst>(*U))
    {
      llvm::Type *src_type = bitcast->getSrcTy();
      llvm::Type *dst_type = bitcast->getDestTy();
      if (!src_type->isPointerTy() || !dst_type->isPointerTy())
        continue;

      // Skip if address space is already the same
      unsigned srcAddrSpace = src_type->getPointerAddressSpace();
      unsigned dstAddrSpace = dst_type->getPointerAddressSpace();
      if (srcAddrSpace == dstAddrSpace)
        continue;

      // Create and insert new bitcast instruction
      llvm::Type *new_dst_type =
        dst_type->getPointerElementType()->getPointerTo(srcAddrSpace);
      llvm::BitCastInst *new_bitcast =
        new llvm::BitCastInst(inst, new_dst_type);
      new_bitcast->insertAfter(bitcast);
      bitcast->replaceAllUsesWith(new_bitcast);
      bitcast->eraseFromParent();

      pocl_update_users_address_space(new_bitcast);
    }
    else if (auto gep = llvm::dyn_cast<llvm::GetElementPtrInst>(*U))
    {
      // Create and insert new GEP instruction
      auto ops = gep->operands();
      std::vector<llvm::Value*> indices(ops.begin()+1, ops.end());
      llvm::GetElementPtrInst *new_gep =
        llvm::GetElementPtrInst::Create(NULL, inst, indices);
      new_gep->insertAfter(gep);
      gep->replaceAllUsesWith(new_gep);
      gep->eraseFromParent();

      pocl_update_users_address_space(new_gep);
    }
  }
}

void pocl_fix_constant_address_space(llvm::Module *module)
{
  // Loop over functions
  std::vector<llvm::Function*> functions;
  for (auto F = module->begin(); F != module->end(); F++)
  {
    functions.push_back(&*F);
  }

  for (auto F = functions.begin(); F != functions.end(); F++)
  {
    // Argument info for creating new function
    std::vector<llvm::Argument*> arguments;
    std::vector<llvm::Type*> argument_types;

    // Loop over arguments
    bool has_constant_args = false;
    for (auto arg = (*F)->arg_begin(); arg != (*F)->arg_end(); arg++)
    {
      // Check for constant memory pointer
      llvm::Type *arg_type = arg->getType();
      if (arg_type->isPointerTy() &&
          arg_type->getPointerAddressSpace() == 4)
      {
        has_constant_args = true;

        // Create new argument in global address space
        llvm::Type *elem_type = arg_type->getPointerElementType();
        llvm::Type *new_arg_type = elem_type->getPointerTo(1);
        llvm::Argument *new_arg = new llvm::Argument(new_arg_type);
        arguments.push_back(new_arg);
        argument_types.push_back(new_arg_type);

        new_arg->takeName(&*arg);
        arg->replaceAllUsesWith(new_arg);

        pocl_update_users_address_space(new_arg);
      }
      else
      {
        // No change to other arguments
        arguments.push_back(&*arg);
        argument_types.push_back(arg_type);
      }
    }

    if (!has_constant_args)
      continue;

    // Create new function with updated arguments
    llvm::FunctionType *new_func_type =
      llvm::FunctionType::get((*F)->getReturnType(), argument_types, false);
    llvm::Function *new_func =
      llvm::Function::Create(new_func_type, (*F)->getLinkage(),
                             (*F)->getName(), module);
    new_func->takeName(&*(*F));

    // Take function body from old function
    new_func->getBasicBlockList().splice(new_func->begin(),
                                         (*F)->getBasicBlockList());

    // TODO: Copy attributes from old function

    // Update function body with new arguments
    std::vector<llvm::Argument*>::iterator old_arg;
    llvm::Function::arg_iterator new_arg;
    for (old_arg = arguments.begin(), new_arg = new_func->arg_begin();
         new_arg != new_func->arg_end();
         new_arg++, old_arg++)
    {
      new_arg->takeName(*old_arg);
      (*old_arg)->replaceAllUsesWith(&*new_arg);
    }

    // Remove old function
    (*F)->replaceAllUsesWith(new_func);
    (*F)->eraseFromParent();
  }
}

void pocl_gen_local_mem_args(llvm::Module *module)
{
  // TODO: Deal with non-kernel functions that take local memory arguments

  llvm::LLVMContext& context = llvm::getGlobalContext();

  llvm::NamedMDNode *md_kernels = module->getNamedMetadata("opencl.kernels");
  if (!md_kernels)
    return;

  // Create global variable for local memory allocations
  llvm::Type *byte_array_type =
    llvm::ArrayType::get(llvm::Type::getInt8Ty(context), 0);
  llvm::GlobalVariable *shared_base =
    new llvm::GlobalVariable(*module, byte_array_type,
                             false, llvm::GlobalValue::ExternalLinkage,
                             NULL, "_shared_memory_region_", NULL,
                             llvm::GlobalValue::NotThreadLocal,
                             3, false);

  // Loop over kernels
  for (auto K = md_kernels->op_begin(); K != md_kernels->op_end(); K++)
  {
    if (!(*K)->getOperand(0))
      continue;

    llvm::ConstantAsMetadata *cam =
      llvm::dyn_cast<llvm::ConstantAsMetadata>((*K)->getOperand(0).get());
    if (!cam)
      continue;

    llvm::Function *function = llvm::dyn_cast<llvm::Function>(cam->getValue());

    // Argument info for creating new function
    std::vector<llvm::Argument*> arguments;
    std::vector<llvm::Type*> argument_types;

    // Loop over arguments
    bool has_local_args = false;
    for (auto arg = function->arg_begin(); arg != function->arg_end(); arg++)
    {
      // Check for local memory pointer
      llvm::Type *arg_type = arg->getType();
      if (arg_type->isPointerTy() &&
          arg_type->getPointerAddressSpace() == 3)
      {
        has_local_args = true;

        // Create new argument for offset into shared memory allocation
        llvm::Type *i32ty = llvm::Type::getInt32Ty(context);
        llvm::Argument *offset =
          new llvm::Argument(i32ty, arg->getName() + "_offset");
        arguments.push_back(offset);
        argument_types.push_back(i32ty);

        // Insert GEP to add offset
        llvm::Value *zero = llvm::ConstantInt::getSigned(i32ty, 0);
        llvm::GetElementPtrInst *gep =
          llvm::GetElementPtrInst::Create(byte_array_type, shared_base,
                                          {zero, offset});
        gep->insertBefore(&*function->begin()->begin());

        // Cast pointer to correct type
        llvm::BitCastInst *cast = new llvm::BitCastInst(gep, arg_type);
        cast->insertAfter(gep);

        cast->takeName(&*arg);
        arg->replaceAllUsesWith(cast);
      }
      else
      {
        // No change to other arguments
        arguments.push_back(&*arg);
        argument_types.push_back(arg_type);
      }
    }

    if (!has_local_args)
      continue;

    // Create new function with offsets instead of local memory pointers
    llvm::FunctionType *new_func_type =
      llvm::FunctionType::get(function->getReturnType(), argument_types, false);
    llvm::Function *new_func =
      llvm::Function::Create(new_func_type, function->getLinkage(),
                             function->getName(), module);
    new_func->takeName(function);

    // Take function body from old function
    new_func->getBasicBlockList().splice(new_func->begin(),
                                         function->getBasicBlockList());

    // TODO: Copy attributes from old function

    // Update function body with new arguments
    std::vector<llvm::Argument*>::iterator old_arg;
    llvm::Function::arg_iterator new_arg;
    for (old_arg = arguments.begin(), new_arg = new_func->arg_begin();
         new_arg != new_func->arg_end();
         new_arg++, old_arg++)
    {
      new_arg->takeName(*old_arg);
      (*old_arg)->replaceAllUsesWith(&*new_arg);
    }

    // Remove old function
    function->replaceAllUsesWith(new_func);
    function->eraseFromParent();
  }
}

void pocl_insert_ptx_intrinsics(llvm::Module *module)
{
  struct ptx_intrinsic_map_entry
  {
    const char *varname;
    const char *intrinsic;
  };
  struct ptx_intrinsic_map_entry intrinsic_map[] =
  {
    {"_local_id_x", "llvm.nvvm.read.ptx.sreg.tid.x"},
    {"_local_id_y", "llvm.nvvm.read.ptx.sreg.tid.y"},
    {"_local_id_z", "llvm.nvvm.read.ptx.sreg.tid.z"},
    {"_local_size_x", "llvm.nvvm.read.ptx.sreg.ntid.x"},
    {"_local_size_y", "llvm.nvvm.read.ptx.sreg.ntid.y"},
    {"_local_size_z", "llvm.nvvm.read.ptx.sreg.ntid.z"},
    {"_group_id_x", "llvm.nvvm.read.ptx.sreg.ctaid.x"},
    {"_group_id_y", "llvm.nvvm.read.ptx.sreg.ctaid.y"},
    {"_group_id_z", "llvm.nvvm.read.ptx.sreg.ctaid.z"},
    {"_num_groups_x", "llvm.nvvm.read.ptx.sreg.nctaid.x"},
    {"_num_groups_y", "llvm.nvvm.read.ptx.sreg.nctaid.y"},
    {"_num_groups_z", "llvm.nvvm.read.ptx.sreg.nctaid.z"},
  };
  size_t num_intrinsics = sizeof(intrinsic_map)/sizeof(ptx_intrinsic_map_entry);

  llvm::LLVMContext& context = llvm::getGlobalContext();
  llvm::Type *int32Ty = llvm::Type::getInt32Ty(context);
  llvm::FunctionType *intrinsic_type = llvm::FunctionType::get(int32Ty, false);

  for (unsigned i = 0; i < num_intrinsics; i++)
  {
    ptx_intrinsic_map_entry entry = intrinsic_map[i];

    llvm::GlobalVariable *var = module->getGlobalVariable(entry.varname);
    if (!var)
      continue;

    auto var_users = var->users();
    std::vector<llvm::Value*> users(var_users.begin(), var_users.end());
    for (auto U = users.begin(); U != users.end(); U++)
    {
      // Look for loads from the global variable
      llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(*U);
      if (load)
      {
        // Replace load with intrinsic
        llvm::Constant *func =
          module->getOrInsertFunction(entry.intrinsic, intrinsic_type);
        llvm::CallInst *call = llvm::CallInst::Create(func, {}, load);
        load->replaceAllUsesWith(call);
        load->eraseFromParent();
      }
    }

    var->eraseFromParent();
  }
}
