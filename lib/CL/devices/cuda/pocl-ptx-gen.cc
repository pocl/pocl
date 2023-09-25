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
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#ifndef LLVM_OLDER_THAN_14_0
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#ifndef LLVM_OLDER_THAN_11_0
#include "llvm/Support/Alignment.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <llvm/PassInfo.h>
#include <llvm/PassRegistry.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>

#include <set>
#include <optional>

namespace llvm {
extern ModulePass *createNVVMReflectPass(const StringMap<int> &Mapping);
}

typedef std::map<std::string, std::vector<size_t>> AlignmentMapT;

static void addKernelAnnotations(llvm::Module *Module);
static void fixConstantMemArgs(llvm::Module *Module);
static void fixLocalMemArgs(llvm::Module *Module);
static void fixPrintF(llvm::Module *Module);
static void handleGetWorkDim(llvm::Module *Module);
static int linkLibDevice(llvm::Module *Module, const char *LibDevicePath);
static void mapLibDeviceCalls(llvm::Module *Module);
static void createAlignmentMap(llvm::Module *Module,
                               AlignmentMapT *AlignmentMap);

namespace pocl {
extern bool isGVarUsedByFunction(llvm::GlobalVariable *GVar, llvm::Function *F);
extern llvm::ModulePass *
createAutomaticLocalsPass(pocl_autolocals_to_args_strategy autolocals_to_args);
extern bool isKernelToProcess(const llvm::Function &F);
} // namespace pocl

static bool verifyModule(llvm::Module *Module, const char *step) {
  std::string Error;
  llvm::raw_string_ostream Errs(Error);
  if (llvm::verifyModule(*Module, &Errs)) {
    POCL_MSG_ERR("[CUDA] ptx-gen step %s: module verification FAILED\n%s\n",
                 step, Error.c_str());
    return false;
  } else {
    // POCL_MSG_PRINT_CUDA("[CUDA] ptx-gen: STEP %s: module verification
    // PASSED\n", step);
    return true;
  }
}

int pocl_ptx_gen(void *llvm_module, const char *PTXFilename, const char *Arch,
                 const char *LibDevicePath, int HasOffsets,
                 void **AlignmentMapPtr) {

  llvm::Module *Module = (llvm::Module *)llvm_module;
  if (!Module) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return CL_BUILD_PROGRAM_FAILURE;
  }
  bool VerifyMod =
      pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT);

  AlignmentMapT *A = new AlignmentMapT;
  assert(*AlignmentMapPtr == nullptr);
  *AlignmentMapPtr = A;
  createAlignmentMap(Module, A);
  if (VerifyMod && !verifyModule(Module, "getAlignmentMap"))
    return CL_BUILD_PROGRAM_FAILURE;

  // Apply transforms to prepare for lowering to PTX.
  fixPrintF(Module);
  if (VerifyMod && !verifyModule(Module, "fixPrintF"))
    return CL_BUILD_PROGRAM_FAILURE;

  fixConstantMemArgs(Module);
  if (VerifyMod && !verifyModule(Module, "fixConstantMemArgs"))
    return CL_BUILD_PROGRAM_FAILURE;

  fixLocalMemArgs(Module);
  if (VerifyMod && !verifyModule(Module, "fixLocalMemArgs"))
    return CL_BUILD_PROGRAM_FAILURE;

  handleGetWorkDim(Module);
  if (VerifyMod && !verifyModule(Module, "handleGetWorkDim"))
    return CL_BUILD_PROGRAM_FAILURE;

  addKernelAnnotations(Module);
  if (VerifyMod && !verifyModule(Module, "addAnnotations"))
    return CL_BUILD_PROGRAM_FAILURE;

  mapLibDeviceCalls(Module);
  if (VerifyMod && !verifyModule(Module, "mapLibDeviceCalls"))
    return CL_BUILD_PROGRAM_FAILURE;

  if (linkLibDevice(Module, LibDevicePath) != 0)
    return CL_BUILD_PROGRAM_FAILURE;
  if (VerifyMod && !verifyModule(Module, "linkLibDevice"))
    return CL_BUILD_PROGRAM_FAILURE;

  if (pocl_get_bool_option("POCL_CUDA_DUMP_NVVM", 0)) {
    std::string ModuleString;
    llvm::raw_string_ostream ModuleStringStream(ModuleString);
    Module->print(ModuleStringStream, NULL);
    POCL_MSG_PRINT_INFO("NVVM module:\n%s\n", ModuleString.c_str());
  }

#ifdef LLVM_OLDER_THAN_11_0
  llvm::StringRef Triple =
      (sizeof(void *) == 8) ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda";
#else
  std::string Triple =
      (sizeof(void *) == 8) ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda";
#endif

  std::string Error;
  // Get NVPTX target.
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget(Triple, Error);

  if (!Target) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to get target\n");
    POCL_MSG_ERR("%s\n", Error.c_str());
    return CL_BUILD_PROGRAM_FAILURE;
  }

  // TODO: Set options?
  llvm::TargetOptions Options;

  // TODO: CPU and features?
#ifdef LLVM_OLDER_THAN_16_0
  std::unique_ptr<llvm::TargetMachine> Machine(
      Target->createTargetMachine(Triple, Arch, "+ptx40", Options, llvm::None));
#else
  std::unique_ptr<llvm::TargetMachine> Machine(
      Target->createTargetMachine(Triple, Arch, "+ptx40", Options, std::nullopt));
#endif
  llvm::legacy::PassManager Passes;

  // Add pass to emit PTX.
  llvm::SmallVector<char, 4096> Data;
  llvm::raw_svector_ostream PTXStream(Data);
  if (Machine->addPassesToEmitFile(Passes, PTXStream,
                                   nullptr,
                                   llvm::CGFT_AssemblyFile)) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to add passes\n");
    return CL_BUILD_PROGRAM_FAILURE;
  }

  // Run passes.
  Passes.run(*Module);

#ifdef LLVM_OLDER_THAN_11_0
  std::string PTX = PTXStream.str();
  const char *Content = PTX.c_str();
  size_t ContentSize = PTX.size();
#else
  llvm::StringRef PTX = PTXStream.str();
  const char *Content = PTX.data();
  size_t ContentSize = PTX.size();
#endif

  if (pocl_write_file(PTXFilename, Content, ContentSize, 0, 0)) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to write final PTX into %s\n",
                 PTXFilename);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  return CL_SUCCESS;
}

// Add the metadata needed to mark a function as a kernel in PTX.
void addKernelAnnotations(llvm::Module *Module) {
  llvm::LLVMContext &Context = Module->getContext();
  llvm::Constant *One =
      llvm::ConstantInt::getSigned(llvm::Type::getInt32Ty(Context), 1);

  // Remove existing nvvm.annotations metadata since it is sometimes corrupt.
  auto *Annotations = Module->getNamedMetadata("nvvm.annotations");
  if (Annotations)
    Annotations->eraseFromParent();

  // Add nvvm.annotations metadata to mark kernel entry point.
  Annotations = Module->getOrInsertNamedMetadata("nvvm.annotations");

  for (auto &FI : Module->functions()) {
    if (!pocl::isKernelToProcess(FI))
      continue;

    llvm::Function *Function = &FI;

    // Create metadata.
    llvm::Metadata *FuncMD = llvm::ValueAsMetadata::get(Function);
    llvm::Metadata *NameMD = llvm::MDString::get(Context, "kernel");
    llvm::Metadata *OneMD = llvm::ConstantAsMetadata::get(One);

    llvm::MDNode *Node = llvm::MDNode::get(Context, {FuncMD, NameMD, OneMD});
    Annotations->addOperand(Node);
  }
}

// PTX doesn't support variadic functions, so we need to modify the IR to
// support printf. The vprintf system call that is provided is described here:
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
// Essentially, the variadic list of arguments is replaced with a single array
// instead.
//
// This function changes the prototype of printf to take an array instead
// of a variadic argument list. It updates the function body to read from
// this array to retrieve each argument instead of using the dummy __cl_va_arg
// function. We then visit each printf callsite and generate the argument
// array to pass instead of the variadic list.
void fixPrintF(llvm::Module *Module) {
  llvm::Function *OldPrintF = Module->getFunction("printf");
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

  // Create new non-variadic printf function.
  llvm::Type *ReturnType = OldPrintF->getReturnType();
  llvm::FunctionType *NewPrintfType =
      llvm::FunctionType::get(ReturnType, {FormatType, I64Ptr}, false);
  llvm::Function *NewPrintF = llvm::Function::Create(
      NewPrintfType, OldPrintF->getLinkage(), "", Module);
  NewPrintF->takeName(OldPrintF);

  // Take function body from old function.
#ifdef LLVM_OLDER_THAN_16_0
  NewPrintF->getBasicBlockList().splice(NewPrintF->begin(),
                                        OldPrintF->getBasicBlockList());
#else
  NewPrintF->splice(NewPrintF->begin(), OldPrintF);
#endif

  // Create i32 to hold current argument index.
#ifdef LLVM_OLDER_THAN_11_0
  llvm::AllocaInst *ArgIndexPtr =
      new llvm::AllocaInst(I32, 0, llvm::ConstantInt::get(I32, 1));
  ArgIndexPtr->insertBefore(&*NewPrintF->begin()->begin());
#else
  llvm::AllocaInst *ArgIndexPtr =
      new llvm::AllocaInst(I32, 0, llvm::ConstantInt::get(I32, 1), llvm::Align(4));
  ArgIndexPtr->insertBefore(&*NewPrintF->begin()->begin());
#endif

#ifdef LLVM_OLDER_THAN_11_0
  llvm::StoreInst *ArgIndexInit =
      new llvm::StoreInst(llvm::ConstantInt::get(I32, 0), ArgIndexPtr);
  ArgIndexInit->insertAfter(ArgIndexPtr);
#else
  llvm::StoreInst *ArgIndexInit =
      new llvm::StoreInst(llvm::ConstantInt::get(I32, 0), ArgIndexPtr, false, llvm::Align(4));
  ArgIndexInit->insertAfter(ArgIndexPtr);
#endif

  // Replace calls to _cl_va_arg with reads from new i64 array argument.
  llvm::Function *VaArgFunc = Module->getFunction("__cl_va_arg");
  if (VaArgFunc) {
    auto args = NewPrintF->arg_begin();
    args++;
    llvm::Argument *ArgsIn = args;
    std::vector<llvm::Value *> VaArgCalls(VaArgFunc->user_begin(),
                                          VaArgFunc->user_end());
    for (auto &U : VaArgCalls) {
      llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
      if (!Call)
        continue;

      // Get current argument index.
#ifdef LLVM_OLDER_THAN_11_0
      llvm::LoadInst *ArgIndex = new llvm::LoadInst(ArgIndexPtr);
      ArgIndex->insertBefore(Call);
#else
      llvm::LoadInst *ArgIndex = new llvm::LoadInst(I32, ArgIndexPtr, "poclCudaLoad", false, llvm::Align(4));
      ArgIndex->insertBefore(Call);
#endif
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
#ifdef LLVM_OLDER_THAN_11_0
      llvm::LoadInst *ArgValue = new llvm::LoadInst(ArgIn);
      ArgValue->insertAfter(ArgIn);
      llvm::StoreInst *ArgStore = new llvm::StoreInst(ArgValue, ArgOut);
      ArgStore->insertAfter(ArgOutBC);
#else
      llvm::LoadInst *ArgValue = new llvm::LoadInst(I64, ArgIn, "poclCudaArgLoad", false, llvm::Align(8));
      ArgValue->insertAfter(ArgIn);
      llvm::StoreInst *ArgStore = new llvm::StoreInst(ArgValue, ArgOut, false, llvm::Align(8));
      ArgStore->insertAfter(ArgOutBC);
#endif
      // Increment argument index.
      llvm::BinaryOperator *Inc = llvm::BinaryOperator::Create(
          llvm::BinaryOperator::Add, ArgIndex, llvm::ConstantInt::get(I32, 1));
      Inc->insertAfter(ArgIndex);

#ifdef LLVM_OLDER_THAN_11_0
      llvm::StoreInst *StoreInc = new llvm::StoreInst(Inc, ArgIndexPtr);
      StoreInc->insertAfter(Inc);
#else
      llvm::StoreInst *StoreInc = new llvm::StoreInst(Inc, ArgIndexPtr, false, llvm::Align(4));
      StoreInc->insertAfter(Inc);
#endif
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

    unsigned NumArgs = Call->arg_size() - 1;

    llvm::Value *Format = Call->getArgOperand(0);

    // Allocate array for arguments.
    // TODO: Deal with vector arguments.
#ifdef LLVM_OLDER_THAN_11_0
    llvm::AllocaInst *Args =
        new llvm::AllocaInst(I64, 0, llvm::ConstantInt::get(I32, NumArgs));
    Args->insertBefore(Call);
#else
    llvm::AllocaInst *Args =
        new llvm::AllocaInst(I64, 0, llvm::ConstantInt::get(I32, NumArgs), llvm::Align(8));
    Args->insertBefore(Call);
#endif

    // Loop over arguments (skipping format).
    for (unsigned A = 0; A < NumArgs; A++) {
      llvm::Value *Arg = Call->getArgOperand(A + 1);
      llvm::Type *ArgType = Arg->getType();

      // Cast pointers to the generic address space.
      if (ArgType->isPointerTy() && ArgType->getPointerAddressSpace() != 0) {
#ifdef LLVM_OPAQUE_POINTERS
        llvm::CastInst *AddrSpaceCast = llvm::CastInst::CreatePointerCast(
            Arg, llvm::PointerType::get(Context, 0));
#else
        llvm::CastInst *AddrSpaceCast =llvm::CastInst::CreatePointerCast(
          Arg, ArgType->getPointerElementType()->getPointerTo());
#endif
        AddrSpaceCast->insertBefore(Call);
        Arg = AddrSpaceCast;
        ArgType = Arg->getType();
      }

      // Get pointer to argument in i64 array.
      // TODO: promote arguments that are shorter than 32 bits.
      llvm::Constant *ArgIndex = llvm::ConstantInt::get(I32, A);
      llvm::Instruction *ArgPtr =
          llvm::GetElementPtrInst::Create(I64, Args, {ArgIndex});
      ArgPtr->insertBefore(Call);

#ifndef LLVM_OPAQUE_POINTERS
      // Cast pointer to correct type if necessary.
      if (ArgPtr->getType()->getPointerElementType() != ArgType) {
        llvm::BitCastInst *ArgPtrBC =
            new llvm::BitCastInst(ArgPtr, ArgType->getPointerTo(0));
        ArgPtrBC->insertAfter(ArgPtr);
        ArgPtr = ArgPtrBC;
      }
#endif
      // Store argument to i64 array.
#ifdef LLVM_OLDER_THAN_11_0
      llvm::StoreInst *Store = new llvm::StoreInst(Arg, ArgPtr);
      Store->insertBefore(Call);
#else
      llvm::StoreInst *Store = new llvm::StoreInst(Arg, ArgPtr, false, llvm::Align(8));
      Store->insertBefore(Call);
#endif
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
#ifdef LLVM_OPAQUE_POINTERS
      llvm::Type *NewFormatType = llvm::PointerType::get(Context, 0);
#else
      llvm::Type *NewFormatType =
          FormatType->getPointerElementType()->getPointerTo(0);
#endif
      llvm::AddrSpaceCastInst *FormatASC =
          new llvm::AddrSpaceCastInst(Format, NewFormatType);
      FormatASC->insertBefore(Call);
      Call->setArgOperand(0, FormatASC);
    }
    Call->setCalledFunction(NewVPrintF);
  }

  VPrintF->eraseFromParent();
}

// TODO broken, replaces in whole module not just  1 function
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
void handleGetWorkDim(llvm::Module *Module) {

  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;

  auto WorkDimVar = Module->getGlobalVariable("_work_dim");
  if (WorkDimVar == nullptr)
    return;

  for (auto &FI : Module->functions()) {
    if (!pocl::isKernelToProcess(FI))
      continue;

    llvm::Function *Function = &FI;
    if (!pocl::isGVarUsedByFunction(WorkDimVar, Function))
      continue;

    // Add additional argument for the work item dimensionality.
    llvm::FunctionType *FunctionType = Function->getFunctionType();
    std::vector<llvm::Type *> ArgumentTypes(FunctionType->param_begin(),
                                            FunctionType->param_end());
    ArgumentTypes.push_back(llvm::Type::getInt32Ty(Module->getContext()));

    // Create new function.
    llvm::FunctionType *NewFunctionType = llvm::FunctionType::get(
        Function->getReturnType(), ArgumentTypes, false);
    llvm::Function *NewFunction = llvm::Function::Create(
        NewFunctionType, Function->getLinkage(), Function->getName(), Module);
    NewFunction->takeName(Function);

    // Map function arguments.
    llvm::ValueToValueMapTy VV;
    llvm::Function::arg_iterator OldArg;
    llvm::Function::arg_iterator NewArg;
    for (OldArg = Function->arg_begin(), NewArg = NewFunction->arg_begin();
         OldArg != Function->arg_end(); NewArg++, OldArg++) {
      NewArg->takeName(&*OldArg);
      VV[&*OldArg] = &*NewArg;
    }

    // Clone function.
    llvm::SmallVector<llvm::ReturnInst *, 1> RI;
    CloneFunctionIntoAbs(NewFunction, Function, VV, RI);
    FunctionsToErase.push_back(Function);

    // Replace uses of the global offset variables with the new arguments.
    NewArg->setName("work_dim");
    // replaceScalarGlobalVar(Module, "_work_dim", (&*NewArg++));

    // TODO: What if get_work_dim() is called from a non-kernel function?
  }

  for (auto F : FunctionsToErase) {
    F->eraseFromParent();
  }
}

int findLibDevice(char LibDevicePath[PATH_MAX], const char *Arch) {
  // Extract numeric portion of SM version.
  char *End;
  unsigned long SM = strtoul(Arch + 3, &End, 10);
  if (!SM || strlen(End)) {
    POCL_MSG_ERR("[CUDA] invalid GPU architecture %s\n", Arch);
    return 1;
  }

  // This mapping from SM version to libdevice library version is given here:
  // http://docs.nvidia.com/cuda/libdevice-users-guide/basic-usage.html#version-selection
  // This is no longer needed as of CUDA 9.
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
    pocl_get_string_option("CUDA_HOME", "/usr/local/cuda"),
    "/usr/local/lib/cuda",
    "/usr/local/lib",
    "/usr/lib",
  };

  static const char *NVVMPath[] = {
    "/nvvm",
    "/nvidia-cuda-toolkit",
    "",
  };

  static const char *PathFormat = "%s%s/libdevice/libdevice.10.bc";
  static const char *OldPathFormat =
      "%s%s/libdevice/libdevice.compute_%d.10.bc";

  // Search combinations of paths for the libdevice library.
  for (auto bp : BasePath) {
    for (auto np : NVVMPath) {
      // Check for CUDA 9+ libdevice library.
      size_t ps = snprintf(LibDevicePath, PATH_MAX - 1, PathFormat, bp, np);
      LibDevicePath[ps] = '\0';
      POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                      "looking for libdevice at '%s'\n", LibDevicePath);
      if (pocl_exists(LibDevicePath)) {
        POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                        "found libdevice at '%s'\n", LibDevicePath);
        return 0;
      }

      // Check for pre CUDA 9 libdevice library.
      ps = snprintf(LibDevicePath, PATH_MAX - 1, OldPathFormat, bp, np,
                    LibDeviceSM);
      LibDevicePath[ps] = '\0';
      POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                      "looking for libdevice at '%s'\n", LibDevicePath);
      if (pocl_exists(LibDevicePath)) {
        POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                        "found libdevice at '%s'\n", LibDevicePath);
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
int linkLibDevice(llvm::Module *Module, const char *LibDevicePath) {
  auto Buffer = llvm::MemoryBuffer::getFile(LibDevicePath);
  if (!Buffer) {
    POCL_MSG_ERR("[CUDA] failed to open libdevice library file\n");
    return -1;
  }

  POCL_MSG_PRINT_INFO("loading libdevice from '%s'\n", LibDevicePath);

  // Load libdevice bitcode library.
  llvm::Expected<std::unique_ptr<llvm::Module>> LibDeviceModule =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), Module->getContext());
  if (!LibDeviceModule) {
    POCL_MSG_ERR("[CUDA] failed to load libdevice bitcode\n");
    return -1;
  }

  // Fix triple and data-layout of libdevice module.
  (*LibDeviceModule)->setTargetTriple(Module->getTargetTriple());
  (*LibDeviceModule)->setDataLayout(Module->getDataLayout());

  // Link libdevice into module.
  llvm::Linker Linker(*Module);
  if (Linker.linkInModule(std::move(LibDeviceModule.get()),
                          llvm::Linker::Flags::LinkOnlyNeeded)) {
    POCL_MSG_ERR("[CUDA] failed to link to libdevice\n");
    return -1;
  }

  llvm::legacy::PassManager Passes;

  // Run internalize to mark all non-kernel functions as internal.
  auto PreserveKernel = [=](const llvm::GlobalValue &GV) {
    const llvm::Function *F = llvm::dyn_cast<llvm::Function>(&GV);
    return (F != nullptr && pocl::isKernelToProcess(*F));
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
  return 0;
}

// This transformation replaces each pointer argument in the specific address
// space with an integer offset, and then inserts the necessary GEP+BitCast
// instructions to calculate the new pointers from the provided base global
// variable.
bool convertPtrArgsToOffsets(llvm::Module *Module, llvm::Function *Function,
                             unsigned AddrSpace, llvm::GlobalVariable *Base) {

  llvm::LLVMContext &Context = Module->getContext();

  // Argument info for creating new function.
  std::vector<llvm::Argument *> Arguments;
  std::vector<llvm::Type *> ArgumentTypes;

  llvm::ValueToValueMapTy VV;
  std::vector<std::pair<llvm::Instruction *, llvm::Instruction *>> ToInsert;

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
#ifdef LLVM_OPAQUE_POINTERS
      llvm::GetElementPtrInst *GEP = llvm::GetElementPtrInst::Create(
          Base->getValueType(), Base, {Zero, Offset});
#else
      llvm::GetElementPtrInst *GEP =
          llvm::GetElementPtrInst::Create(Base->getType()->getPointerElementType(), Base, {Zero, Offset});
#endif
      // Cast pointer to correct type.
      llvm::BitCastInst *Cast = new llvm::BitCastInst(GEP, ArgType);

      // Save these instructions to insert into new function later.
      ToInsert.push_back({GEP, Cast});

      // Map the old local memory argument to the result of this cast.
      VV[&Arg] = Cast;

    } else {
      // No change to other arguments.
      Arguments.push_back(&Arg);
      ArgumentTypes.push_back(ArgType);
    }
  }

  if (!NeedsArgOffsets)
    return false;

  // Create new function with offsets instead of local memory pointers.
  llvm::FunctionType *NewFunctionType =
      llvm::FunctionType::get(Function->getReturnType(), ArgumentTypes, false);
  llvm::Function *NewFunction = llvm::Function::Create(
      NewFunctionType, Function->getLinkage(), Function->getName(), Module);
  NewFunction->takeName(Function);

  // Map function arguments.
  std::vector<llvm::Argument *>::iterator OldArg;
  llvm::Function::arg_iterator NewArg;
  for (OldArg = Arguments.begin(), NewArg = NewFunction->arg_begin();
       NewArg != NewFunction->arg_end(); NewArg++, OldArg++) {
    NewArg->takeName(*OldArg);
    if ((*OldArg)->getParent())
      VV[*OldArg] = &*NewArg;
    else {
      // Manually replace new offset arguments.
      (*OldArg)->replaceAllUsesWith(&*NewArg);
      delete *OldArg;
    }
  }

  // Clone function.
  llvm::SmallVector<llvm::ReturnInst *, 1> RI;
  CloneFunctionIntoAbs(NewFunction, Function, VV, RI);

  // Insert offset instructions into new function.
  for (auto Pair : ToInsert) {
    Pair.first->insertBefore(&*NewFunction->begin()->begin());
    Pair.second->insertAfter(Pair.first);
  }

  return true;
}

// CUDA doesn't allow constant pointer arguments, so we have to convert them to
// offsets and manually add them to a global variable base pointer.
void fixConstantMemArgs(llvm::Module *Module) {

  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;

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
      *Module, ByteArrayType, false, llvm::GlobalValue::ExternalLinkage,
      NULL, "_constant_memory_region_",
      NULL, llvm::GlobalValue::NotThreadLocal, 4, false);

  for (auto &FI : Module->functions()) {
    if (!pocl::isKernelToProcess(FI))
      continue;
    if (convertPtrArgsToOffsets(Module, &FI, 4, ConstantMemBase))
      FunctionsToErase.push_back(&FI);
  }
  for (auto F : FunctionsToErase) {
    F->eraseFromParent();
  }
}

// CUDA doesn't allow multiple local memory arguments or automatic variables, so
// we have to create a single global variable for local memory allocations, and
// then manually add offsets to it to get each individual local memory
// allocation.
void fixLocalMemArgs(llvm::Module *Module) {

  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;

  // Create global variable for local memory allocations.
  llvm::Type *ByteArrayType =
      llvm::ArrayType::get(llvm::Type::getInt8Ty(Module->getContext()), 0);
  llvm::GlobalVariable *SharedMemBase = new llvm::GlobalVariable(
      *Module, ByteArrayType, false, llvm::GlobalValue::ExternalLinkage, NULL,
      "_shared_memory_region_", NULL, llvm::GlobalValue::NotThreadLocal, 3,
      false);

  for (auto &FI : Module->functions()) {
    if (!pocl::isKernelToProcess(FI))
      continue;
    if (convertPtrArgsToOffsets(Module, &FI, 3, SharedMemBase))
      FunctionsToErase.push_back(&FI);
  }

  for (auto F : FunctionsToErase) {
    F->eraseFromParent();
  }
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
  {        name "f",    "__nv_" name "f"}, \
  {        name,        "__nv_" name}, \
  {"llvm." name ".f32", "__nv_" name "f"}, \
  {"llvm." name ".f64", "__nv_" name},

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

    {"llvm.powi.f32", "__nv_powif"},
    {"llvm.powi.f64", "__nv_powi"},

    {"frexp", "__nv_frexp"},
    {"frexpf", "__nv_frexpf"},

    {"tgamma", "__nv_tgamma"},
    {"tgammaf", "__nv_tgammaf"},

    {"ldexp", "__nv_ldexp"},
    {"ldexpf", "__nv_ldexpf"},

    {"modf", "__nv_modf"},
    {"modff", "__nv_modff"},

    {"remquo", "__nv_remquo"},
    {"remquof", "__nv_remquof"},

    // TODO: lgamma_r
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
        llvm::FunctionCallee FC = Module->getOrInsertFunction(
            Entry.LibDeviceFunctionName, FunctionType);
        llvm::Function *LibDeviceFunction = llvm::cast<llvm::Function>(FC.getCallee());
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

static int getPtrArgAlignment(llvm::Module *Module, llvm::Function *Kernel,
                              std::vector<size_t> &AlignmentVec) {
  // Load the LLVM bitcode module.
  if (!Module) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return -1;
  }

  // Get kernel function.
  if (!Kernel) {
    POCL_MSG_ERR("[CUDA] kernel function not found in module\n");
    return -1;
  }

  // Calculate alignment for each argument.
  const llvm::DataLayout &DL = Module->getDataLayout();
  for (auto &Arg : Kernel->args()) {
    unsigned i = Arg.getArgNo();
    llvm::Type *Type = Arg.getType();
    AlignmentVec.push_back(0);
    assert(i < AlignmentVec.size());
    // TODO test this
    if (Type->isPointerTy()) {
      // try to figure out alignment from uses
      for (auto U : Arg.users()) {
        if (llvm::GetElementPtrInst *GEP =
                llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
          for (auto UU : GEP->users()) {
            if (llvm::StoreInst *SI = llvm::dyn_cast<llvm::StoreInst>(UU)) {
              AlignmentVec[i] = SI->getAlign().value();
              break;
            }
            if (llvm::LoadInst *LI = llvm::dyn_cast<llvm::LoadInst>(UU)) {
              AlignmentVec[i] = LI->getAlign().value();
              break;
            }
          }
          if (AlignmentVec[i])
            break;
        }
      }
      if (AlignmentVec[i] == 0)
        AlignmentVec[i] = MAX_EXTENDED_ALIGNMENT;
      //      POCL_MSG_WARN("V1 ||||| Argument %u : ALIGN %zu \n", i,
      //      AlignmentVec[i]);
    } else {
#ifdef LLVM_OLDER_THAN_16_0
      AlignmentVec[i] = Arg.getParamAlignment();
#else
      if (Arg.getType()->isPointerTy())
        AlignmentVec[i] = Arg.getParamAlign().valueOrOne().value();
      else
        AlignmentVec[i] = DL.getTypeAllocSize(Arg.getType());
#endif
      //      POCL_MSG_WARN("V2 ||||| Argument %u : ALIGN %zu \n", i,
      //      AlignmentVec[i]);
    }
  }

  return 0;
}

void createAlignmentMap(llvm::Module *Module, AlignmentMapT *AlignmentMap) {
  for (auto &FI : Module->functions()) {
    if (!pocl::isKernelToProcess(FI))
      continue;
    std::string Name = FI.getName().str();
    if (AlignmentMap->find(Name) != AlignmentMap->end())
      continue;
    std::vector<size_t> &AVec = (*AlignmentMap)[Name];
    AVec.reserve(4);
    if (getPtrArgAlignment(Module, &FI, AVec) != 0)
      POCL_MSG_ERR("can't figure out alignments for kernel %s", Name.c_str());
  }
}

int pocl_cuda_create_alignments(void *llvm_module, void **AlignmentMapPtr) {
  llvm::Module *Module = (llvm::Module *)llvm_module;
  if (!Module) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed to load bitcode\n");
    return -1;
  }
  bool VerifyMod =
      pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT);

  AlignmentMapT *A = new AlignmentMapT;
  assert(*AlignmentMapPtr == nullptr);
  *AlignmentMapPtr = A;
  createAlignmentMap(Module, A);

  if (VerifyMod && !verifyModule(Module, "getAlignmentMap"))
    return -1;
  return 0;
}

void pocl_cuda_destroy_alignments(void *llvm_module, void *AlignmentMapPtr) {
  if (AlignmentMapPtr != nullptr) {
    AlignmentMapT *A = (AlignmentMapT *)(AlignmentMapPtr);
    delete A;
  }
}

int pocl_cuda_get_ptr_arg_alignment(void *LLVM_IR, const char *KernelName,
                                    size_t *Alignments, void *AlignmentMapPtr) {
  AlignmentMapT *AMap = (AlignmentMapT *)AlignmentMapPtr;
  assert(AMap != nullptr);

  std::string Name(KernelName);
  if (AMap->find(Name) == AMap->end()) {
    POCL_MSG_ERR(
        "pocl_cuda_get_ptr_arg_alignment: kernel not found in module\n");
    return 1;
  }
  std::vector<size_t> &AVec = AMap->at(Name);
  //  POCL_MSG_WARN("AVEC SIZE: %zu\n", AVec.size());
  std::memcpy(Alignments, AVec.data(), sizeof(size_t) * AVec.size());
  return 0;
}
