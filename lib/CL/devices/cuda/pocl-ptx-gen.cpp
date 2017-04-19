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
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"
#include "pocl-ptx-gen.h"

#ifdef LLVM_OLDER_THAN_4_0
#include "llvm/Bitcode/ReaderWriter.h"
#else
#include "llvm/Bitcode/BitcodeReader.h"
#endif
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <set>

namespace llvm
{
  extern ModulePass* createNVVMReflectPass(const StringMap<int>& Mapping);
}

// TODO: Should these be proper passes?
void pocl_add_kernel_annotations(llvm::Module *module, const char *kernel);
void pocl_cuda_fix_printf(llvm::Module *module);
void pocl_cuda_link_libdevice(llvm::Module *module,
                              const char *kernel, const char *gpu_arch);
void pocl_gen_local_mem_args(llvm::Module *module, const char *kernel);
void pocl_insert_ptx_intrinsics(llvm::Module *module);
void pocl_map_libdevice_calls(llvm::Module *module);

int pocl_ptx_gen(const char *bc_filename,
                 const char *ptx_filename,
                 const char *kernel_name,
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
  llvm::LLVMContext context;
#ifdef LLVM_OLDER_THAN_4_0
  llvm::ErrorOr<std::unique_ptr<llvm::Module>> module =
#else
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
#endif
    parseBitcodeFile(buffer->get()->getMemBufferRef(), context);
  if (!module)
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return 1;
  }

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  // Apply transforms to prepare for lowering to PTX
  pocl_cuda_fix_printf(module->get());
  pocl_gen_local_mem_args(module->get(), kernel_name);
  pocl_insert_ptx_intrinsics(module->get());
  pocl_add_kernel_annotations(module->get(), kernel_name);
  pocl_map_libdevice_calls(module->get());
  pocl_cuda_link_libdevice(module->get(), kernel_name, gpu_arch);
  if (pocl_get_bool_option("POCL_DEBUG_PTX", 0))
    (*module)->dump();

  // Verify module
  std::string err;
  llvm::raw_string_ostream errs(err);
  if (llvm::verifyModule(*module->get(), &errs))
  {
    printf("\n%s\n", err.c_str());
    POCL_ABORT("[CUDA] ptx-gen: module verification failed\n");
  }

  llvm::StringRef triple =
    (sizeof(void*)==8) ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda";

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
#ifdef LLVM_OLDER_THAN_3_9
      target->createTargetMachine(triple, gpu_arch, "+ptx40", options));
#else
      target->createTargetMachine(triple, gpu_arch, "+ptx40", options, llvm::None));
#endif

  llvm::legacy::PassManager passes;

  // Add pass to emit PTX
  llvm::SmallVector<char, 4096> data;
  llvm::raw_svector_ostream ptx_stream(data);
  if (machine->addPassesToEmitFile(passes, ptx_stream,
                                   llvm::TargetMachine::CGFT_AssemblyFile))
  {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to add passes\n");
    return 1;
  }

  // Run passes
  passes.run(**module);

  std::string ptx = ptx_stream.str(); // flush
  return pocl_write_file(ptx_filename, ptx.c_str(), ptx.size(), 0, 0);
}

void pocl_add_kernel_annotations(llvm::Module *module, const char *kernel)
{
  llvm::LLVMContext& context = module->getContext();

  // Remove existing nvvm.annotations metadata since it is sometimes corrupt
  auto nvvm_annotations = module->getNamedMetadata("nvvm.annotations");
  if (nvvm_annotations)
    nvvm_annotations->eraseFromParent();

  // Add nvvm.annotations metadata to mark kernel entry point
  nvvm_annotations = module->getOrInsertNamedMetadata("nvvm.annotations");

  // Get handle to function
  auto func = module->getFunction(kernel);
  if (!func)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Create metadata
  llvm::Constant *one =
    llvm::ConstantInt::getSigned(llvm::Type::getInt32Ty(context), 1);
  llvm::Metadata *md_f = llvm::ValueAsMetadata::get(func);
  llvm::Metadata *md_n = llvm::MDString::get(context, "kernel");
  llvm::Metadata *md_1 = llvm::ConstantAsMetadata::get(one);

  std::vector<llvm::Metadata*> v_md = {md_f, md_n, md_1};
  llvm::MDNode *node = llvm::MDNode::get(context, v_md);
  nvvm_annotations->addOperand(node);
}

void pocl_erase_function_and_callers(llvm::Function *func)
{
  if (!func)
    return;

  std::vector<llvm::Value*> callers(func->users().begin(), func->users().end());
  for (auto U = callers.begin(); U != callers.end(); U++)
  {
    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*U);
    if (!call)
      continue;
    call->eraseFromParent();
  }
  func->eraseFromParent();
}

void pocl_cuda_fix_printf(llvm::Module *module)
{
  llvm::Function *cl_printf = module->getFunction("__cl_printf");
  if (!cl_printf)
    return;

  llvm::LLVMContext& context = module->getContext();
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *i64ptr = llvm::PointerType::get(i64, 0);
  llvm::Type *format_type = cl_printf->getFunctionType()->getParamType(0);

  // Remove calls to va_start and va_end
  pocl_erase_function_and_callers(module->getFunction("llvm.va_start"));
  pocl_erase_function_and_callers(module->getFunction("llvm.va_end"));

  // Create new non-variadic __cl_printf function
  llvm::Type *ret_type = cl_printf->getReturnType();
  llvm::FunctionType *new_func_type =
    llvm::FunctionType::get(ret_type, {format_type, i64ptr}, false);
  llvm::Function *new_cl_printf =
    llvm::Function::Create(new_func_type, cl_printf->getLinkage(), "", module);
  new_cl_printf->takeName(cl_printf);

  // Take function body from old function
  new_cl_printf->getBasicBlockList().splice(new_cl_printf->begin(),
                                            cl_printf->getBasicBlockList());

  // Create i32 to hold current argument index
  llvm::AllocaInst *arg_index_ptr =
    new llvm::AllocaInst(i32, llvm::ConstantInt::get(i32, 1));
  arg_index_ptr->insertBefore(&*new_cl_printf->begin()->begin());
  llvm::StoreInst *arg_index_init =
    new llvm::StoreInst(llvm::ConstantInt::get(i32, 0), arg_index_ptr);
  arg_index_init->insertAfter(arg_index_ptr);

  // Replace calls to _cl_va_arg with reads from new i64 array argument
  llvm::Function *cl_va_arg = module->getFunction("__cl_va_arg");
  if (cl_va_arg)
  {
    llvm::Argument *args_in = &*++new_cl_printf->getArgumentList().begin();
    std::vector<llvm::Value*> va_arg_calls(cl_va_arg->users().begin(),
                                           cl_va_arg->users().end());
    for (auto U = va_arg_calls.begin(); U != va_arg_calls.end(); U++)
    {
      llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*U);
      if (!call)
        continue;

      // Get current argument index
      llvm::LoadInst *arg_index = new llvm::LoadInst(arg_index_ptr);
      arg_index->insertBefore(call);

      // Get pointer to argument data
      llvm::Value *arg_out = call->getArgOperand(1);
      llvm::GetElementPtrInst *arg_in =
        llvm::GetElementPtrInst::Create(i64, args_in, {arg_index});
      arg_in->insertAfter(arg_index);

      // Cast arg_out pointer to i64*
      llvm::BitCastInst *bc_arg_out = new llvm::BitCastInst(arg_out, i64ptr);
      bc_arg_out->insertAfter(arg_in);
      arg_out = bc_arg_out;

      // Load argument
      llvm::LoadInst *arg_value = new llvm::LoadInst(arg_in);
      arg_value->insertAfter(arg_in);
      llvm::StoreInst *arg_store = new llvm::StoreInst(arg_value, arg_out);
      arg_store->insertAfter(bc_arg_out);

      // Increment argument index
      llvm::BinaryOperator *inc =
        llvm::BinaryOperator::Create(llvm::BinaryOperator::Add,
                                     arg_index, llvm::ConstantInt::get(i32,1));
      inc->insertAfter(arg_index);
      llvm::StoreInst *store_inc = new llvm::StoreInst(inc, arg_index_ptr);
      store_inc->insertAfter(inc);

      // Remove call to _cl_va_arg
      call->eraseFromParent();
    }

    // Remove function from module
    cl_va_arg->eraseFromParent();
  }

  // Loop over function callers
  // Generate array of i64 arguments to replace variadic arguments
  std::vector<llvm::Value*> callers(cl_printf->users().begin(),
                                    cl_printf->users().end());
  for (auto U = callers.begin(); U != callers.end(); U++)
  {
    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*U);
    if (!call)
      continue;

    unsigned num_args = call->getNumArgOperands() - 1;
    llvm::Value *format = call->getArgOperand(0);

    // Allocate array for arguments
    // TODO: Deal with vector arguments
    llvm::AllocaInst *args =
      new llvm::AllocaInst(i64, llvm::ConstantInt::get(i32, num_args));
    args->insertBefore(call);

    // Loop over arguments (skipping format)
    for (unsigned a = 0; a < num_args; a++)
    {
      llvm::Value *arg = call->getArgOperand(a+1);
      llvm::Type *arg_type = arg->getType();

      // Get pointer to argument in i64 array
      // TODO: promote arguments that are shorter than 32 bits
      llvm::Constant *arg_idx = llvm::ConstantInt::get(i32, a);
      llvm::Instruction *arg_ptr =
        llvm::GetElementPtrInst::Create(i64, args, {arg_idx});
      arg_ptr->insertBefore(call);

      // Cast pointer to correct type if necessary
      if (arg_ptr->getType()->getPointerElementType() != arg_type)
      {
        llvm::BitCastInst *bc_arg_ptr =
          new llvm::BitCastInst(arg_ptr, arg_type->getPointerTo(0));
        bc_arg_ptr->insertAfter(arg_ptr);
        arg_ptr = bc_arg_ptr;
      }

      // Store argument to i64 array
      llvm::StoreInst *store = new llvm::StoreInst(arg, arg_ptr);
      store->insertBefore(call);
    }

    // Fix address space of undef format values
    if (format->getValueID() == llvm::Value::UndefValueVal)
    {
      format = llvm::UndefValue::get(format_type);
    }

    // Replace call with new non-variadic function
    llvm::CallInst *new_call =
      llvm::CallInst::Create(new_cl_printf, {format, args});
    new_call->insertBefore(call);
    call->replaceAllUsesWith(new_call);
    call->eraseFromParent();
  }

  // Update arguments
  llvm::Function::arg_iterator old_arg = cl_printf->arg_begin();
  llvm::Function::arg_iterator new_arg = new_cl_printf->arg_begin();
  new_arg->takeName(&*old_arg);
  old_arg->replaceAllUsesWith(&*new_arg);

  // Remove old function
  cl_printf->eraseFromParent();


  // Get handle to vprintf function
  llvm::Function *vprintf_func = module->getFunction("vprintf");
  if (!vprintf_func)
    return;

  // If vprintf format address space is already generic, then we're done
  auto vprintf_format_type = vprintf_func->getFunctionType()->getParamType(0);
  if (vprintf_format_type->getPointerAddressSpace() == 0)
    return;

  // Change address space of vprintf format argument to generic
  auto i8ptr = llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0);
  auto new_vprintf_type = llvm::FunctionType::get(vprintf_func->getReturnType(),
                                                  {i8ptr, i8ptr}, false);
  auto new_vprintf = llvm::Function::Create(new_vprintf_type,
                                            vprintf_func->getLinkage(),
                                            "", module);
  new_vprintf->takeName(vprintf_func);

  // Update vprintf callers to pass format arguments in generic address space
  callers.assign(vprintf_func->users().begin(), vprintf_func->users().end());
  for (auto U = callers.begin(); U != callers.end(); U++)
  {
    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*U);
    if (!call)
      continue;

    llvm::Value *format = call->getArgOperand(0);
    llvm::Type *arg_type = format->getType();
    if (arg_type->getPointerAddressSpace() != 0)
    {
      // Cast address space to generic
      llvm::Type *new_arg_type =
        arg_type->getPointerElementType()->getPointerTo(0);
      llvm::AddrSpaceCastInst *asc =
        new llvm::AddrSpaceCastInst(format, new_arg_type);
      asc->insertBefore(call);
      call->setArgOperand(0, asc);
    }
    call->setCalledFunction(new_vprintf);
  }

  vprintf_func->eraseFromParent();
}

void pocl_cuda_link_libdevice(llvm::Module *module,
                              const char *kernel, const char *gpu_arch)
{
  // TODO: Can we link libdevice into the kernel library at pocl build time?
  // This would remove this runtime depenency on the CUDA toolkit.
  // Had some issues with the other pocl LLVM passess crashing on the libdevice
  // code - needs more investigation.

  // Construct path to libdevice bitcode library
  const char *cuda_path = pocl_get_string_option("CUDA_PATH", "");
  const char *libdevice_fmt = "%s/nvvm/libdevice/libdevice.compute_%d.10.bc";

  char *end;
  unsigned long sm = strtoul(gpu_arch+3, &end, 10);
  if (!sm || strlen(end))
    POCL_ABORT("[CUDA] invalid GPU architecture %s\n", gpu_arch);

  // This mapping from SM version to libdevice library version is given here:
  // http://docs.nvidia.com/cuda/libdevice-users-guide/basic-usage.html#version-selection
  int ldsm = 0;
  if (sm < 30)
    ldsm = 20;
  else if (sm == 30)
    ldsm = 30;
  else if (sm < 35)
    ldsm = 20;
  else if (sm <= 37)
    ldsm = 35;
  else if (sm < 50)
    ldsm = 30;
  else if (sm <= 53)
    ldsm = 50;
  else
    ldsm = 30;

  size_t sz = snprintf(NULL, 0, libdevice_fmt, cuda_path, ldsm);
  char *libdevice_path = (char*)malloc(sz + 1);
  sprintf(libdevice_path, libdevice_fmt, cuda_path, ldsm);
  POCL_MSG_PRINT_INFO("loading libdevice from '%s'\n", libdevice_path);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
    llvm::MemoryBuffer::getFile(libdevice_path);
  free(libdevice_path);
  if (!buffer)
    POCL_ABORT("[CUDA] failed to open libdevice library file\n");

  // Load libdevice bitcode library
#ifdef LLVM_OLDER_THAN_4_0
  llvm::ErrorOr<std::unique_ptr<llvm::Module>> libdevice_module =
#else
  llvm::Expected<std::unique_ptr<llvm::Module>> libdevice_module =
#endif
    parseBitcodeFile(buffer->get()->getMemBufferRef(), module->getContext());
  if (!libdevice_module)
    POCL_ABORT("[CUDA] failed to load libdevice bitcode\n");

  // Fix triple and data-layout of libdevice module
  (*libdevice_module)->setTargetTriple(module->getTargetTriple());
  (*libdevice_module)->setDataLayout(module->getDataLayout());

  // Link libdevice into module
  llvm::Linker linker(*module);
  if (linker.linkInModule(std::move(libdevice_module.get())))
  {
    POCL_ABORT("[CUDA] failed to link to libdevice");
  }


  llvm::legacy::PassManager passes;

  // Run internalize to mark all non-kernel functions as internal
#ifdef LLVM_OLDER_THAN_3_9
  passes.add(llvm::createInternalizePass({kernel}));
#else
  auto preserve_kernel = [=](const llvm::GlobalValue &GV)
  {
    return GV.getName() == kernel;
  };
  passes.add(llvm::createInternalizePass(preserve_kernel));
#endif

  // Run NVVM reflect pass to set math options
  // TODO: Determine correct FTZ value from frontend compiler options
  llvm::StringMap<int> reflect_params;
  reflect_params["__CUDA_FTZ"] = 1;
  passes.add(llvm::createNVVMReflectPass(reflect_params));

  // Run optimization passes to clean up unused functions etc
  llvm::PassManagerBuilder Builder;
  Builder.OptLevel = 3;
  Builder.SizeLevel = 0;
  Builder.populateModulePassManager(passes);

  passes.run(*module);
}

void pocl_gen_local_mem_args(llvm::Module *module, const char *kernel)
{
  // TODO: Deal with non-kernel functions that take local memory arguments

  llvm::LLVMContext& context = module->getContext();

  llvm::Function *function = module->getFunction(kernel);
  if (!function)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Create global variable for local memory allocations
  llvm::Type *byte_array_type =
    llvm::ArrayType::get(llvm::Type::getInt8Ty(context), 0);
  llvm::GlobalVariable *shared_base =
    new llvm::GlobalVariable(*module, byte_array_type,
                             false, llvm::GlobalValue::ExternalLinkage,
                             NULL, "_shared_memory_region_", NULL,
                             llvm::GlobalValue::NotThreadLocal,
                             3, false);


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
    return;

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

  // TODO: Deal with calls to this kernel from other function?

  function->eraseFromParent();
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

  llvm::LLVMContext& context = module->getContext();
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
        llvm::CallInst *call = llvm::CallInst::Create(func, "", load);
        load->replaceAllUsesWith(call);
        load->eraseFromParent();
      }
    }

    var->eraseFromParent();
  }
}

void pocl_map_libdevice_calls(llvm::Module *module)
{
  struct function_map_entry
  {
    const char *ocl_funcname;
    const char *libdevice_funcname;
  };
  struct function_map_entry function_map[] =
  {
    // TODO: FP64 versions - not all of them just worked in the same way...

#define LDMAP(name) \
  {name "f", "__nv_" name "f"}, \
  {name,     "__nv_" name},
    LDMAP("acos")
    LDMAP("acosh")
    LDMAP("asin")
    LDMAP("asinh")
    LDMAP("atan")
    LDMAP("atanh")
    LDMAP("atan2")
    LDMAP("cbrt")
    LDMAP("ceil")
    LDMAP("copysign")
    LDMAP("cos")
    LDMAP("cosh")
    LDMAP("exp")
    LDMAP("exp2")
    LDMAP("expm1")
    LDMAP("fdim")
    LDMAP("floor")
    LDMAP("fmax")
    LDMAP("fmin")
    LDMAP("hypot")
    LDMAP("ilogb")
    LDMAP("lgamma")
    LDMAP("log")
    LDMAP("log2")
    LDMAP("log10")
    LDMAP("log1p")
    LDMAP("logb")
    LDMAP("nextafter")
    LDMAP("remainder")
    LDMAP("rint")
    LDMAP("round")
    LDMAP("sin")
    LDMAP("sinh")
    LDMAP("sqrt")
    LDMAP("tan")
    LDMAP("tanh")
    LDMAP("trunc")
#undef LDMAP

    {"llvm.pow.f32", "__nv_powf"},
    {"llvm.pow.f64", "__nv_pow"},

    // TODO: frexp
    // TODO: ldexp
    // TODO: lgamma_r
    // TODO: modf
    // TODO: pown
    // TODO: remquo
    // TODO: rootn
  };
  size_t num_functions = sizeof(function_map)/sizeof(function_map_entry);

  for (unsigned i = 0; i < num_functions; i++)
  {
    function_map_entry entry = function_map[i];

    llvm::Function *func = module->getFunction(entry.ocl_funcname);
    if (!func)
      continue;

    auto func_users = func->users();
    std::vector<llvm::Value*> users(func_users.begin(), func_users.end());
    for (auto U = users.begin(); U != users.end(); U++)
    {
      // Look for calls to function
      llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*U);
      if (call)
      {
        // Create function declaration for lidevice version
        llvm::FunctionType *func_type = func->getFunctionType();
        llvm::Constant *libdevice_func =
          module->getOrInsertFunction(entry.libdevice_funcname, func_type);

        // Replace function with libdevice version
        std::vector<llvm::Value*> args;
        for (auto arg = call->arg_begin(); arg != call->arg_end(); arg++)
          args.push_back(*arg);
        llvm::CallInst *new_call =
          llvm::CallInst::Create(libdevice_func, args, "", call);
        new_call->takeName(call);
        call->replaceAllUsesWith(new_call);
        call->eraseFromParent();
      }
    }

    func->eraseFromParent();
  }
}
