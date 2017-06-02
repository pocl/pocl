/* pocl-ptx-gen.cc - PTX code generation functions

   Copyright (c) 2016-2017 James Price / University of Bristol

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

#include "LLVMUtils.h"
#include "common.h"
#include "pocl-ptx-gen.h"
#include "pocl.h"
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
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

namespace llvm {
extern ModulePass *createNVVMReflectPass(const StringMap<int> &Mapping);
}

static void addKernelAnnotations(llvm::Module *Module, const char *KernelName);
static void fixConstantMemArgs(llvm::Module *Module, const char *KernelName);
static void fixLocalMemArgs(llvm::Module *Module, const char *KernelName);
static void fixPrintF(llvm::Module *Module);
static void handleGetWorkDim(llvm::Module *Module, const char *KernelName);
static void handleGlobalOffsets(llvm::Module *Module, const char *KernelName,
                                bool HasOffsets);
static void linkLibDevice(llvm::Module *Module, const char *KernelName,
                          const char *LibDevicePath);
static void mapLibDeviceCalls(llvm::Module *Module);

int pocl_ptx_gen(const char *BitcodeFilename, const char *PTXFilename,
                 const char *KernelName, const char *Arch,
                 const char *LibDevicePath, int HasOffsets) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(BitcodeFilename);
  if (!Buffer) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to open bitcode file\n");
    return 1;
  }

  // Load the LLVM bitcode module.
  llvm::LLVMContext Context;
  llvm::Expected<std::unique_ptr<llvm::Module>> Module =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), Context);
  if (!Module) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return 1;
  }

  // Apply transforms to prepare for lowering to PTX.
  fixPrintF(Module->get());
  fixConstantMemArgs(Module->get(), KernelName);
  fixLocalMemArgs(Module->get(), KernelName);
  handleGetWorkDim(Module->get(), KernelName);
  handleGlobalOffsets(Module->get(), KernelName, HasOffsets);
  addKernelAnnotations(Module->get(), KernelName);
  mapLibDeviceCalls(Module->get());
  linkLibDevice(Module->get(), KernelName, LibDevicePath);
  if (pocl_get_bool_option("POCL_CUDA_DUMP_NVVM", 0)) {
    std::string ModuleString;
    llvm::raw_string_ostream ModuleStringStream(ModuleString);
    (*Module)->print(ModuleStringStream, NULL);
    POCL_MSG_PRINT_INFO("NVVM module:\n%s\n", ModuleString.c_str());
  }

  // Verify module.
  std::string Error;
  llvm::raw_string_ostream Errs(Error);
  if (llvm::verifyModule(*Module->get(), &Errs)) {
    POCL_MSG_ERR("\n%s\n", Error.c_str());
    POCL_ABORT("[CUDA] ptx-gen: module verification failed\n");
  }

  llvm::StringRef Triple =
      (sizeof(void *) == 8) ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda";

  // Get NVPTX target.
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget(Triple, Error);
  if (!Target) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to get target\n");
    POCL_MSG_ERR("%s\n", Error.c_str());
    return 1;
  }

  // TODO: Set options?
  llvm::TargetOptions Options;

  // TODO: CPU and features?
  std::unique_ptr<llvm::TargetMachine> Machine(
      Target->createTargetMachine(Triple, Arch, "+ptx40", Options, llvm::None));

  llvm::legacy::PassManager Passes;

  // Add pass to emit PTX.
  llvm::SmallVector<char, 4096> Data;
  llvm::raw_svector_ostream PTXStream(Data);
  if (Machine->addPassesToEmitFile(Passes, PTXStream,
                                   llvm::TargetMachine::CGFT_AssemblyFile)) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to add passes\n");
    return 1;
  }

  // Run passes.
  Passes.run(**Module);

  std::string PTX = PTXStream.str();
  return pocl_write_file(PTXFilename, PTX.c_str(), PTX.size(), 0, 0);
}

// Add the metadata needed to mark a function as a kernel in PTX.
void addKernelAnnotations(llvm::Module *Module, const char *KernelName) {
  llvm::LLVMContext &Context = Module->getContext();

  // Remove existing nvvm.annotations metadata since it is sometimes corrupt.
  auto *Annotations = Module->getNamedMetadata("nvvm.annotations");
  if (Annotations)
    Annotations->eraseFromParent();

  // Add nvvm.annotations metadata to mark kernel entry point.
  Annotations = Module->getOrInsertNamedMetadata("nvvm.annotations");

  // Get handle to function.
  auto *Function = Module->getFunction(KernelName);
  if (!Function)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Create metadata.
  llvm::Constant *One =
      llvm::ConstantInt::getSigned(llvm::Type::getInt32Ty(Context), 1);
  llvm::Metadata *FuncMD = llvm::ValueAsMetadata::get(Function);
  llvm::Metadata *NameMD = llvm::MDString::get(Context, "kernel");
  llvm::Metadata *OneMD = llvm::ConstantAsMetadata::get(One);

  llvm::MDNode *Node = llvm::MDNode::get(Context, {FuncMD, NameMD, OneMD});
  Annotations->addOperand(Node);
}

// PTX doesn't support variadic functions, so we need to modify the IR to
// support printf. The vprintf system call that is provided is described here:
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
// Essentially, the variadic list of arguments is replaced with a single array
// instead.
//
// This function changes the prototype of __cl_printf to take an array instead
// of a variadic argument list. It updates the function body to read from
// this array to retrieve each argument instead of using the dummy __cl_va_arg
// function. We then visit each __cl_printf callsite and generate the argument
// array to pass instead of the variadic list.
void fixPrintF(llvm::Module *Module) {
  llvm::Function *OldPrintF = Module->getFunction("__cl_printf");
  if (!OldPrintF)
    return;

  llvm::LLVMContext &Context = Module->getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Context);
  llvm::Type *I64 = llvm::Type::getInt64Ty(Context);
  llvm::Type *I64Ptr = llvm::PointerType::get(I64, 0);
  llvm::Type *FormatType = OldPrintF->getFunctionType()->getParamType(0);

  // Remove calls to va_start and va_end.
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_start"));
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_end"));

  // Create new non-variadic __cl_printf function.
  llvm::Type *ReturnType = OldPrintF->getReturnType();
  llvm::FunctionType *NewPrintfType =
      llvm::FunctionType::get(ReturnType, {FormatType, I64Ptr}, false);
  llvm::Function *NewPrintF = llvm::Function::Create(
      NewPrintfType, OldPrintF->getLinkage(), "", Module);
  NewPrintF->takeName(OldPrintF);

  // Take function body from old function.
  NewPrintF->getBasicBlockList().splice(NewPrintF->begin(),
                                        OldPrintF->getBasicBlockList());

  // Create i32 to hold current argument index.
  llvm::AllocaInst *ArgIndexPtr =
#if LLVM_OLDER_THAN_5_0
      new llvm::AllocaInst(I32, llvm::ConstantInt::get(I32, 1));
#else
      new llvm::AllocaInst(I32, 0, llvm::ConstantInt::get(I32, 1));
#endif
  ArgIndexPtr->insertBefore(&*NewPrintF->begin()->begin());
  llvm::StoreInst *ArgIndexInit =
      new llvm::StoreInst(llvm::ConstantInt::get(I32, 0), ArgIndexPtr);
  ArgIndexInit->insertAfter(ArgIndexPtr);

  // Replace calls to _cl_va_arg with reads from new i64 array argument.
  llvm::Function *VaArgFunc = Module->getFunction("__cl_va_arg");
  if (VaArgFunc) {
#if LLVM_OLDER_THAN_5_0
    llvm::Argument *ArgsIn = &*++NewPrintF->arg_begin();
#else
    auto args = NewPrintF->arg_begin();
    args++;
    llvm::Argument *ArgsIn = args;
#endif
    std::vector<llvm::Value *> VaArgCalls(VaArgFunc->user_begin(),
                                          VaArgFunc->user_end());
    for (auto &U : VaArgCalls) {
      llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
      if (!Call)
        continue;

      // Get current argument index.
      llvm::LoadInst *ArgIndex = new llvm::LoadInst(ArgIndexPtr);
      ArgIndex->insertBefore(Call);

      // Get pointer to argument data.
      llvm::Value *ArgOut = Call->getArgOperand(1);
      llvm::GetElementPtrInst *ArgIn =
          llvm::GetElementPtrInst::Create(I64, ArgsIn, {ArgIndex});
      ArgIn->insertAfter(ArgIndex);

      // Cast ArgOut pointer to i64*.
      llvm::BitCastInst *ArgOutBC = new llvm::BitCastInst(ArgOut, I64Ptr);
      ArgOutBC->insertAfter(ArgIn);
      ArgOut = ArgOutBC;

      // Load argument.
      llvm::LoadInst *ArgValue = new llvm::LoadInst(ArgIn);
      ArgValue->insertAfter(ArgIn);
      llvm::StoreInst *ArgStore = new llvm::StoreInst(ArgValue, ArgOut);
      ArgStore->insertAfter(ArgOutBC);

      // Increment argument index.
      llvm::BinaryOperator *Inc = llvm::BinaryOperator::Create(
          llvm::BinaryOperator::Add, ArgIndex, llvm::ConstantInt::get(I32, 1));
      Inc->insertAfter(ArgIndex);
      llvm::StoreInst *StoreInc = new llvm::StoreInst(Inc, ArgIndexPtr);
      StoreInc->insertAfter(Inc);

      // Remove call to _cl_va_arg.
      Call->eraseFromParent();
    }

    // Remove function from module.
    VaArgFunc->eraseFromParent();
  }

  // Loop over function callers.
  // Generate array of i64 arguments to replace variadic arguments/
  std::vector<llvm::Value *> Callers(OldPrintF->user_begin(),
                                     OldPrintF->user_end());
  for (auto &U : Callers) {
    llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
    if (!Call)
      continue;

    unsigned NumArgs = Call->getNumArgOperands() - 1;
    llvm::Value *Format = Call->getArgOperand(0);

    // Allocate array for arguments.
    // TODO: Deal with vector arguments.
    llvm::AllocaInst *Args =
#if LLVM_OLDER_THAN_5_0
        new llvm::AllocaInst(I64, llvm::ConstantInt::get(I32, NumArgs));
#else
        new llvm::AllocaInst(I64, 0, llvm::ConstantInt::get(I32, NumArgs));
#endif
    Args->insertBefore(Call);

    // Loop over arguments (skipping format).
    for (unsigned A = 0; A < NumArgs; A++) {
      llvm::Value *Arg = Call->getArgOperand(A + 1);
      llvm::Type *ArgType = Arg->getType();

      // Get pointer to argument in i64 array.
      // TODO: promote arguments that are shorter than 32 bits.
      llvm::Constant *ArgIndex = llvm::ConstantInt::get(I32, A);
      llvm::Instruction *ArgPtr =
          llvm::GetElementPtrInst::Create(I64, Args, {ArgIndex});
      ArgPtr->insertBefore(Call);

      // Cast pointer to correct type if necessary.
      if (ArgPtr->getType()->getPointerElementType() != ArgType) {
        llvm::BitCastInst *ArgPtrBC =
            new llvm::BitCastInst(ArgPtr, ArgType->getPointerTo(0));
        ArgPtrBC->insertAfter(ArgPtr);
        ArgPtr = ArgPtrBC;
      }

      // Store argument to i64 array.
      llvm::StoreInst *Store = new llvm::StoreInst(Arg, ArgPtr);
      Store->insertBefore(Call);
    }

    // Fix address space of undef format values.
    if (Format->getValueID() == llvm::Value::UndefValueVal) {
      Format = llvm::UndefValue::get(FormatType);
    }

    // Replace call with new non-variadic function.
    llvm::CallInst *NewCall = llvm::CallInst::Create(NewPrintF, {Format, Args});
    NewCall->insertBefore(Call);
    Call->replaceAllUsesWith(NewCall);
    Call->eraseFromParent();
  }

  // Update arguments.
  llvm::Function::arg_iterator OldArg = OldPrintF->arg_begin();
  llvm::Function::arg_iterator NewArg = NewPrintF->arg_begin();
  NewArg->takeName(&*OldArg);
  OldArg->replaceAllUsesWith(&*NewArg);

  // Remove old function.
  OldPrintF->eraseFromParent();

  // Get handle to vprintf function.
  llvm::Function *VPrintF = Module->getFunction("vprintf");
  if (!VPrintF)
    return;

  // If vprintf format address space is already generic, then we're done.
  auto *VPrintFFormatType = VPrintF->getFunctionType()->getParamType(0);
  if (VPrintFFormatType->getPointerAddressSpace() == 0)
    return;

  // Change address space of vprintf format argument to generic.
  auto *I8Ptr = llvm::PointerType::get(llvm::Type::getInt8Ty(Context), 0);
  auto *NewVPrintFType =
      llvm::FunctionType::get(VPrintF->getReturnType(), {I8Ptr, I8Ptr}, false);
  auto *NewVPrintF =
      llvm::Function::Create(NewVPrintFType, VPrintF->getLinkage(), "", Module);
  NewVPrintF->takeName(VPrintF);

  // Update vprintf callers to pass format arguments in generic address space.
  Callers.assign(VPrintF->user_begin(), VPrintF->user_end());
  for (auto &U : Callers) {
    llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
    if (!Call)
      continue;

    llvm::Value *Format = Call->getArgOperand(0);
    llvm::Type *FormatType = Format->getType();
    if (FormatType->getPointerAddressSpace() != 0) {
      // Cast address space to generic.
      llvm::Type *NewFormatType =
          FormatType->getPointerElementType()->getPointerTo(0);
      llvm::AddrSpaceCastInst *FormatASC =
          new llvm::AddrSpaceCastInst(Format, NewFormatType);
      FormatASC->insertBefore(Call);
      Call->setArgOperand(0, FormatASC);
    }
    Call->setCalledFunction(NewVPrintF);
  }

  VPrintF->eraseFromParent();
}

// Replace all load users of a scalar global variable with new value.
static void replaceScalarGlobalVar(llvm::Module *Module, const char *Name,
                                   llvm::Value *NewValue) {
  auto GlobalVar = Module->getGlobalVariable(Name);
  if (!GlobalVar)
    return;

  std::vector<llvm::Value *> Users(GlobalVar->user_begin(),
                                   GlobalVar->user_end());
  for (auto *U : Users) {
    auto Load = llvm::dyn_cast<llvm::LoadInst>(U);
    assert(Load && "Use of a scalar global variable is not a load");
    Load->replaceAllUsesWith(NewValue);
    Load->eraseFromParent();
  }
  GlobalVar->eraseFromParent();
}

// Add an extra kernel argument for the dimensionality.
void handleGetWorkDim(llvm::Module *Module, const char *KernelName) {
  llvm::Function *Function = Module->getFunction(KernelName);
  if (!Function)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Add additional argument for the work item dimensionality.
  llvm::FunctionType *FunctionType = Function->getFunctionType();
  std::vector<llvm::Type *> ArgumentTypes(FunctionType->param_begin(),
                                          FunctionType->param_end());
  ArgumentTypes.push_back(llvm::Type::getInt32Ty(Module->getContext()));

  // Create new function.
  llvm::FunctionType *NewFunctionType =
      llvm::FunctionType::get(Function->getReturnType(), ArgumentTypes, false);
  llvm::Function *NewFunction = llvm::Function::Create(
      NewFunctionType, Function->getLinkage(), Function->getName(), Module);
  NewFunction->takeName(Function);

  // Take function body from old function.
  NewFunction->getBasicBlockList().splice(NewFunction->begin(),
                                          Function->getBasicBlockList());

  // TODO: Copy attributes from old function?

  // Update function body with the original arguments.
  llvm::Function::arg_iterator OldArg;
  llvm::Function::arg_iterator NewArg;
  for (OldArg = Function->arg_begin(), NewArg = NewFunction->arg_begin();
       OldArg != Function->arg_end(); NewArg++, OldArg++) {
    NewArg->takeName(&*OldArg);
    (&*OldArg)->replaceAllUsesWith(&*NewArg);
  }

  Function->eraseFromParent();

  auto WorkDimVar = Module->getGlobalVariable("_work_dim");
  if (!WorkDimVar)
    return;

  // Replace uses of the global offset variables with the new arguments.
  NewArg->setName("work_dim");
  replaceScalarGlobalVar(Module, "_work_dim", (&*NewArg++));

  // TODO: What if get_work_dim() is called from a non-kernel function?
}

// If we don't need to handle offsets, just replaces uses of the offset
// variables with constant zero. Otherwise, add additional kernel arguments for
// the offsets and use those instead.
void handleGlobalOffsets(llvm::Module *Module, const char *KernelName,
                         bool HasOffsets) {
  if (!HasOffsets) {
    llvm::Type *I32 = llvm::Type::getInt32Ty(Module->getContext());
    llvm::Value *Zero = llvm::ConstantInt::getSigned(I32, 0);
    replaceScalarGlobalVar(Module, "_global_offset_x", Zero);
    replaceScalarGlobalVar(Module, "_global_offset_y", Zero);
    replaceScalarGlobalVar(Module, "_global_offset_z", Zero);
    return;
  }

  llvm::Function *Function = Module->getFunction(KernelName);
  if (!Function)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Add additional arguments for the global offsets.
  llvm::FunctionType *FunctionType = Function->getFunctionType();
  std::vector<llvm::Type *> ArgumentTypes(FunctionType->param_begin(),
                                          FunctionType->param_end());
  llvm::Type *I32 = llvm::Type::getInt32Ty(Module->getContext());
  ArgumentTypes.push_back(I32);
  ArgumentTypes.push_back(I32);
  ArgumentTypes.push_back(I32);

  // Create new function.
  llvm::FunctionType *NewFunctionType =
      llvm::FunctionType::get(Function->getReturnType(), ArgumentTypes, false);
  llvm::Function *NewFunction = llvm::Function::Create(
      NewFunctionType, Function->getLinkage(), Function->getName(), Module);
  NewFunction->takeName(Function);

  // Take function body from old function.
  NewFunction->getBasicBlockList().splice(NewFunction->begin(),
                                          Function->getBasicBlockList());

  // TODO: Copy attributes from old function?

  // Update function body with the original arguments.
  llvm::Function::arg_iterator OldArg;
  llvm::Function::arg_iterator NewArg;
  for (OldArg = Function->arg_begin(), NewArg = NewFunction->arg_begin();
       OldArg != Function->arg_end(); NewArg++, OldArg++) {
    NewArg->takeName(&*OldArg);
    (&*OldArg)->replaceAllUsesWith(&*NewArg);
  }

  // Replace uses of the global offset variables with the new arguments.
  NewArg->setName("global_offset_x");
  replaceScalarGlobalVar(Module, "_global_offset_x", (&*NewArg++));
  NewArg->setName("global_offset_y");
  replaceScalarGlobalVar(Module, "_global_offset_y", (&*NewArg++));
  NewArg->setName("global_offset_z");
  replaceScalarGlobalVar(Module, "_global_offset_z", (&*NewArg++));

  // TODO: What if the offsets are in a function that isn't the kernel?

  Function->eraseFromParent();
}

int findLibDevice(char LibDevicePath[PATH_MAX], const char *Arch) {
  // Extract numeric portion of SM version.
  char *End;
  unsigned long SM = strtoul(Arch + 3, &End, 10);
  if (!SM || strlen(End))
    {
      POCL_MSG_ERR ("[CUDA] invalid GPU architecture %s\n", Arch);
      return 1;
    }

  // This mapping from SM version to libdevice library version is given here:
  // http://docs.nvidia.com/cuda/libdevice-users-guide/basic-usage.html#version-selection
  int LibDeviceSM = 0;
  if (SM < 30)
    LibDeviceSM = 20;
  else if (SM == 30)
    LibDeviceSM = 30;
  else if (SM < 35)
    LibDeviceSM = 20;
  else if (SM <= 37)
    LibDeviceSM = 35;
  else if (SM < 50)
    LibDeviceSM = 30;
  else if (SM <= 53)
    LibDeviceSM = 50;
  else
    LibDeviceSM = 30;

  const char *BasePath[] = {
    pocl_get_string_option("POCL_CUDA_TOOLKIT_PATH", CUDA_TOOLKIT_ROOT_DIR),
    "/usr/local/lib/cuda",
    "/usr/local/lib",
    "/usr/lib",
  };

  static const char *NVVMPath[] = {
    "/nvvm",
    "/nvidia-cuda-toolkit",
    "",
  };

  static const char *PathFormat = "%s%s/libdevice/libdevice.compute_%d.10.bc";

  // Search combinations of paths for the libdevice library.
  for (auto bp : BasePath) {
    for (auto np : NVVMPath) {
      size_t ps = snprintf(LibDevicePath, PATH_MAX - 1, PathFormat, bp, np,
                           LibDeviceSM);
      LibDevicePath[ps] = '\0';
      POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__, "looking for libdevice at '%s'\n",
                      LibDevicePath);
      if (pocl_exists(LibDevicePath)) {
        POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__, "found libdevice at '%s'\n",
                        LibDevicePath);
        return 0;
      }
    }
  }

  return 1;
}

// Link CUDA's libdevice bitcode library to provide implementations for most of
// the OpenCL math functions.
// TODO: Can we link libdevice into the kernel library at pocl build time?
// This would remove this runtime dependency on the CUDA toolkit.
// Had some issues with the earlier pocl LLVM passes crashing on the libdevice
// code - needs more investigation.
void linkLibDevice(llvm::Module *Module, const char *KernelName,
                   const char *LibDevicePath) {
  auto Buffer = llvm::MemoryBuffer::getFile(LibDevicePath);
  if (!Buffer)
    POCL_ABORT("[CUDA] failed to open libdevice library file\n");

  POCL_MSG_PRINT_INFO("loading libdevice from '%s'\n", LibDevicePath);

  // Load libdevice bitcode library.
  llvm::Expected<std::unique_ptr<llvm::Module>> LibDeviceModule =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), Module->getContext());
  if (!LibDeviceModule)
    POCL_ABORT("[CUDA] failed to load libdevice bitcode\n");

  // Fix triple and data-layout of libdevice module.
  (*LibDeviceModule)->setTargetTriple(Module->getTargetTriple());
  (*LibDeviceModule)->setDataLayout(Module->getDataLayout());

  // Link libdevice into module.
  llvm::Linker Linker(*Module);
  if (Linker.linkInModule(std::move(LibDeviceModule.get()))) {
    POCL_ABORT("[CUDA] failed to link to libdevice");
  }

  llvm::legacy::PassManager Passes;

  // Run internalize to mark all non-kernel functions as internal.
  auto PreserveKernel = [=](const llvm::GlobalValue &GV) {
    return GV.getName() == KernelName;
  };
  Passes.add(llvm::createInternalizePass(PreserveKernel));

  // Add NVVM reflect module flags to set math options.
  // TODO: Determine correct FTZ value from frontend compiler options.
  llvm::LLVMContext &Context = Module->getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Context);
  llvm::Metadata *FourMD =
      llvm::ValueAsMetadata::get(llvm::ConstantInt::getSigned(I32, 4));
  llvm::Metadata *NameMD = llvm::MDString::get(Context, "nvvm-reflect-ftz");
  llvm::Metadata *OneMD =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 1));
  llvm::MDNode *ReflectFlag =
      llvm::MDNode::get(Context, {FourMD, NameMD, OneMD});
  Module->addModuleFlag(ReflectFlag);

  // Run optimization passes to clean up unused functions etc.
  llvm::PassManagerBuilder Builder;
  Builder.OptLevel = 3;
  Builder.SizeLevel = 0;
  Builder.populateModulePassManager(Passes);

  Passes.run(*Module);
}

// This transformation replaces each pointer argument in the specific address
// space with an integer offset, and then inserts the necessary GEP+BitCast
// instructions to calculate the new pointers from the provided base global
// variable.
void convertPtrArgsToOffsets(llvm::Module *Module, const char *KernelName,
                             unsigned AddrSpace, llvm::GlobalVariable *Base) {

  llvm::LLVMContext &Context = Module->getContext();

  llvm::Function *Function = Module->getFunction(KernelName);
  if (!Function)
    POCL_ABORT("[CUDA] ptx-gen: kernel function not found in module\n");

  // Argument info for creating new function.
  std::vector<llvm::Argument *> Arguments;
  std::vector<llvm::Type *> ArgumentTypes;

  // Loop over arguments.
  bool NeedsArgOffsets = false;
  for (auto &Arg : Function->args()) {
    // Check for local memory pointer.
    llvm::Type *ArgType = Arg.getType();
    if (ArgType->isPointerTy() &&
        ArgType->getPointerAddressSpace() == AddrSpace) {
      NeedsArgOffsets = true;

      // Create new argument for offset into shared memory allocation.
      llvm::Type *I32ty = llvm::Type::getInt32Ty(Context);
      llvm::Argument *Offset =
          new llvm::Argument(I32ty, Arg.getName() + "_offset");
      Arguments.push_back(Offset);
      ArgumentTypes.push_back(I32ty);

      // Insert GEP to add offset.
      llvm::Value *Zero = llvm::ConstantInt::getSigned(I32ty, 0);
      llvm::GetElementPtrInst *GEP =
          llvm::GetElementPtrInst::Create(nullptr, Base, {Zero, Offset});
      GEP->insertBefore(&*Function->begin()->begin());

      // Cast pointer to correct type.
      llvm::BitCastInst *Cast = new llvm::BitCastInst(GEP, ArgType);
      Cast->insertAfter(GEP);

      Cast->takeName(&Arg);
      Arg.replaceAllUsesWith(Cast);
    } else {
      // No change to other arguments.
      Arguments.push_back(&Arg);
      ArgumentTypes.push_back(ArgType);
    }
  }

  if (!NeedsArgOffsets)
    return;

  // Create new function with offsets instead of local memory pointers.
  llvm::FunctionType *NewFunctionType =
      llvm::FunctionType::get(Function->getReturnType(), ArgumentTypes, false);
  llvm::Function *NewFunction = llvm::Function::Create(
      NewFunctionType, Function->getLinkage(), Function->getName(), Module);
  NewFunction->takeName(Function);

  // Take function body from old function.
  NewFunction->getBasicBlockList().splice(NewFunction->begin(),
                                          Function->getBasicBlockList());

  // TODO: Copy attributes from old function.

  // Update function body with new arguments.
  std::vector<llvm::Argument *>::iterator OldArg;
  llvm::Function::arg_iterator NewArg;
  for (OldArg = Arguments.begin(), NewArg = NewFunction->arg_begin();
       NewArg != NewFunction->arg_end(); NewArg++, OldArg++) {
    NewArg->takeName(*OldArg);
    (*OldArg)->replaceAllUsesWith(&*NewArg);
  }

  // TODO: Deal with calls to this kernel from other function?

  Function->eraseFromParent();
}

// CUDA doesn't allow constant pointer arguments, so we have to convert them to
// offsets and manually add them to a global variable base pointer.
void fixConstantMemArgs(llvm::Module *Module, const char *KernelName) {

  // Calculate total size of automatic constant allocations.
  size_t TotalAutoConstantSize = 0;
  for (auto &GlobalVar : Module->globals()) {
    if (GlobalVar.getType()->getPointerAddressSpace() == 4)
      TotalAutoConstantSize += Module->getDataLayout().getTypeAllocSize(
          GlobalVar.getInitializer()->getType());
  }

  // Create global variable for constant memory allocations.
  // TODO: Does allocating the maximum amount have a penalty?
  llvm::Type *ByteArrayType =
      llvm::ArrayType::get(llvm::Type::getInt8Ty(Module->getContext()),
                           65536 - TotalAutoConstantSize);
  llvm::GlobalVariable *ConstantMemBase = new llvm::GlobalVariable(
      *Module, ByteArrayType, false, llvm::GlobalValue::InternalLinkage,
      llvm::Constant::getNullValue(ByteArrayType), "_constant_memory_region_",
      NULL, llvm::GlobalValue::NotThreadLocal, 4, false);

  convertPtrArgsToOffsets(Module, KernelName, 4, ConstantMemBase);
}

// CUDA doesn't allow multiple local memory arguments or automatic variables, so
// we have to create a single global variable for local memory allocations, and
// then manually add offsets to it to get each individual local memory
// allocation.
void fixLocalMemArgs(llvm::Module *Module, const char *KernelName) {

  // Create global variable for local memory allocations.
  llvm::Type *ByteArrayType =
      llvm::ArrayType::get(llvm::Type::getInt8Ty(Module->getContext()), 0);
  llvm::GlobalVariable *SharedMemBase = new llvm::GlobalVariable(
      *Module, ByteArrayType, false, llvm::GlobalValue::ExternalLinkage, NULL,
      "_shared_memory_region_", NULL, llvm::GlobalValue::NotThreadLocal, 3,
      false);

  convertPtrArgsToOffsets(Module, KernelName, 3, SharedMemBase);
}

// Map kernel math functions onto the corresponding CUDA libdevice functions.
void mapLibDeviceCalls(llvm::Module *Module) {
  struct FunctionMapEntry {
    const char *OCLFunctionName;
    const char *LibDeviceFunctionName;
  };
  struct FunctionMapEntry FunctionMap[] = {

// clang-format off
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

    {"llvm.copysign.f32", "__nv_copysignf"},
    {"llvm.copysign.f64", "__nv_copysign"},

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
  // clang-format on

  for (auto &Entry : FunctionMap) {
    llvm::Function *Function = Module->getFunction(Entry.OCLFunctionName);
    if (!Function)
      continue;

    std::vector<llvm::Value *> Users(Function->user_begin(),
                                     Function->user_end());
    for (auto &U : Users) {
      // Look for calls to function.
      llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
      if (Call) {
        // Create function declaration for libdevice version.
        llvm::FunctionType *FunctionType = Function->getFunctionType();
        llvm::Constant *LibDeviceFunction = Module->getOrInsertFunction(
            Entry.LibDeviceFunctionName, FunctionType);

        // Replace function with libdevice version.
        std::vector<llvm::Value *> Args(Call->arg_begin(), Call->arg_end());
        llvm::CallInst *NewCall =
            llvm::CallInst::Create(LibDeviceFunction, Args, "", Call);
        NewCall->takeName(Call);
        Call->replaceAllUsesWith(NewCall);
        Call->eraseFromParent();
      }
    }

    Function->eraseFromParent();
  }
}

int pocl_cuda_get_ptr_arg_alignment(const char *BitcodeFilename,
                                    const char *KernelName,
                                    size_t *Alignments) {
  // Create buffer for bitcode file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(BitcodeFilename);
  if (!Buffer) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to open bitcode file\n");
    return 1;
  }

  // Load the LLVM bitcode module.
  llvm::LLVMContext Context;
  llvm::Expected<std::unique_ptr<llvm::Module>> Module =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), Context);
  if (!Module) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return 1;
  }

  // Get kernel function.
  llvm::Function *Kernel = (*Module)->getFunction(KernelName);
  if (!Kernel)
    POCL_ABORT("[CUDA] kernel function not found in module\n");

  // Calculate alignment for each argument.
  const llvm::DataLayout &DL = (*Module)->getDataLayout();
  for (auto &Arg : Kernel->args()) {
    unsigned i = Arg.getArgNo();
    llvm::Type *Type = Arg.getType();
    if (!Type->isPointerTy())
      Alignments[i] = 0;
    else {
      llvm::Type *ElemType = Type->getPointerElementType();
      Alignments[i] = DL.getTypeAllocSize(ElemType);
    }
  }

  return 0;
}
