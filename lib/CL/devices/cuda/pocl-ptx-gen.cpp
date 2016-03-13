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
void pocl_gen_local_mem_args(llvm::Module *module);
void pocl_insert_ptx_intrinsics(llvm::Module *module);

int pocl_ptx_gen(char *bc_filename, char *ptx_filename)
{
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
    llvm::MemoryBuffer::getFile(bc_filename);
  if (!buffer)
    return 1;

  // Load bitcode
  llvm::ErrorOr<std::unique_ptr<llvm::Module>> module =
    parseBitcodeFile(buffer->get()->getMemBufferRef(),
    llvm::getGlobalContext());
  if (!module)
    return 1;

  // Apply transforms to prepare for lowering to PTX
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
    return 1;

  // TODO: set options
  llvm::TargetOptions options;

  // TODO: CPU and features?
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, "", "", options));

  llvm::legacy::PassManager passes;

  // Add pass to emit PTX
  std::error_code ec;
  llvm::raw_fd_ostream *ptx =
    new llvm::raw_fd_ostream(ptx_filename, ec, llvm::sys::fs::F_Text);
  if (machine->addPassesToEmitFile(passes, *ptx,
                                   llvm::TargetMachine::CGFT_AssemblyFile))
    return 1;

  // Run passes
  passes.run(**module);

  return 0;
}

void pocl_add_kernel_annotations(llvm::Module *module)
{
  llvm::LLVMContext& context = llvm::getGlobalContext();

  // Add nvvm.annotations metadata to mark kernel entry points
  llvm::NamedMDNode *md_kernels = module->getNamedMetadata("opencl.kernels");
  if (!md_kernels)
    return;

  llvm::NamedMDNode *nvvm_annotations =
    module->getOrInsertNamedMetadata("nvvm.annotations");
  for (auto K = md_kernels->op_begin(); K != md_kernels->op_end(); K++)
  {
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

    llvm::ArrayRef<llvm::Metadata*> md({md_f, md_n, md_1});
    nvvm_annotations->addOperand(llvm::MDNode::get(context, md));
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
  llvm::GlobalVariable *shared_ptr =
    new llvm::GlobalVariable(*module, byte_array_type,
                             false, llvm::GlobalValue::ExternalLinkage,
                             NULL, "_shared_memory_region_", NULL,
                             llvm::GlobalValue::NotThreadLocal,
                             3, false);

  // Loop over kernels
  for (auto K = md_kernels->op_begin(); K != md_kernels->op_end(); K++)
  {
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
      if (arg_type->getPointerAddressSpace() == POCL_ADDRESS_SPACE_LOCAL)
      {
        has_local_args = true;

        // Create new argument for offset into shared memory allocation
        llvm::Type *i32ty = llvm::Type::getInt32Ty(context);
        llvm::Argument *offset =
          new llvm::Argument(i32ty, arg->getName() + "_offset");
        arguments.push_back(offset);
        argument_types.push_back(i32ty);

        // Cast shared memory pointer to generic address space
        llvm::AddrSpaceCastInst *generic_ptr =
          new llvm::AddrSpaceCastInst(shared_ptr,
                                      byte_array_type->getPointerTo(0));
        generic_ptr->insertBefore(&*function->begin()->begin());

        // Insert GEP to add offset
        llvm::Value *zero = llvm::ConstantInt::getSigned(i32ty, 0);
        llvm::GetElementPtrInst *gep =
          llvm::GetElementPtrInst::Create(byte_array_type, generic_ptr,
                                          {zero, offset});
        gep->insertAfter(generic_ptr);

        // Cast pointer to correct type
        llvm::Type *final_type =
          arg_type->getPointerElementType()->getPointerTo(0);
        llvm::BitCastInst *cast = new llvm::BitCastInst(gep, final_type);
        cast->insertAfter(gep);

        cast->takeName(&*arg);
        arg->replaceAllUsesWith(cast);

        // Update users of this new cast to use generic address space
        auto cast_users = cast->users();
        std::vector<llvm::Value*> users(cast_users.begin(), cast_users.end());
        for (auto U = users.begin(); U != users.end(); U++)
        {
          // TODO: Do we need to do this for anything other than GEPs?

          llvm::GetElementPtrInst *gep_user =
            llvm::dyn_cast<llvm::GetElementPtrInst>(*U);
          if (!gep_user)
            continue;

          // Create and insert new GEP instruction
          auto ops = gep_user->operands();
          std::vector<llvm::Value*> indices(ops.begin()+1, ops.end());
          llvm::GetElementPtrInst *new_gep_user =
            llvm::GetElementPtrInst::Create(NULL, cast, indices);
          new_gep_user->insertAfter(gep_user);
          gep_user->replaceAllUsesWith(new_gep_user);
          gep_user->eraseFromParent();
        }
      }
      else
      {
        // No change to other arguments
        arguments.push_back(&*arg);
        argument_types.push_back(arg->getType());
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

    for (auto U = var->user_begin(); U != var->user_end(); U++)
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
