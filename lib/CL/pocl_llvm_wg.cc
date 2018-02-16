/* pocl_llvm_wg.cc: part of pocl LLVM API dealing with parallel.bc,
   optimization passes and codegen.

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

#include "config.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_llvm_api.h"
#include "pocl_file_util.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>

#include <llvm/Support/Casting.h>
#include <llvm/Support/MutexGuard.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/CommandLine.h>

#include <llvm/ADT/Triple.h>
#include <llvm/ADT/StringRef.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>

#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>

#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/PassRegistry.h>
#include <llvm/PassInfo.h>

#ifdef LLVM_OLDER_THAN_3_7
#include <llvm/PassManager.h>
#include <llvm/Target/TargetLibraryInfo.h>
#else
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#define PassManager legacy::PassManager
#endif

using namespace llvm;

/**
 * Prepare the kernel compiler passes.
 *
 * The passes are created only once per program run per device.
 * The returned pass manager should not be modified, only the Module
 * should be optimized using it.
 */

static std::map<cl_device_id, llvm::TargetMachine *> targetMachines;
static std::map<cl_device_id, PassManager *> kernelPasses;

// This is used to control the kernel we process in the kernel compilation.
extern cl::opt<std::string> KernelName;

/* FIXME: these options should come from the cl_device, and
 * cl_program's options. */
static llvm::TargetOptions GetTargetOptions() {
  llvm::TargetOptions Options;
#ifdef LLVM_OLDER_THAN_3_9
  Options.PositionIndependentExecutable = true;
#endif
#ifdef HOST_FLOAT_SOFT_ABI
  Options.FloatABIType = FloatABI::Soft;
#else
  Options.FloatABIType = FloatABI::Hard;
#endif
#if 0
  Options.LessPreciseFPMADOption = EnableFPMAD;
  Options.NoFramePointerElim = DisableFPElim;
  Options.NoFramePointerElimNonLeaf = DisableFPElimNonLeaf;
  Options.AllowFPOpFusion = FuseFPOps;
  Options.UnsafeFPMath = EnableUnsafeFPMath;
  Options.NoInfsFPMath = EnableNoInfsFPMath;
  Options.NoNaNsFPMath = EnableNoNaNsFPMath;
  Options.HonorSignDependentRoundingFPMathOption =
  EnableHonorSignDependentRoundingFPMath;
  Options.UseSoftFloat = GenerateSoftFloatCalls;
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;
  Options.NoZerosInBSS = DontPlaceZerosInBSS;
  Options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
  Options.DisableTailCalls = DisableTailCalls;
  Options.StackAlignmentOverride = OverrideStackAlignment;
  Options.RealignStack = EnableRealignStack;
  Options.TrapFuncName = TrapFuncName;
  Options.EnableSegmentedStacks = SegmentedStacks;
  Options.UseInitArray = UseInitArray;
  Options.SSPBufferSize = SSPBufferSize;
#endif
  return Options;
}

void clearTargetMachines() {
  for (auto i = targetMachines.begin(), e = targetMachines.end(); i != e; ++i) {
    delete (llvm::TargetMachine *)i->second;
  }
  targetMachines.clear();
}

void clearKernelPasses() {
  for (auto i = kernelPasses.begin(), e = kernelPasses.end();
       i != e; ++i) {
    PassManager *pm = (PassManager *)i->second;
    delete pm;
  }

  kernelPasses.clear();
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine *GetTargetMachine(cl_device_id device) {

  if (targetMachines.find(device) != targetMachines.end())
    return targetMachines[device];

  std::string Error;
  Triple TheTriple(device->llvm_target_triplet);

  std::string MCPU = device->llvm_cpu ? device->llvm_cpu : "";

  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);

  // In LLVM 3.4 and earlier, the target registry falls back to
  // the cpp backend in case a proper match was not found. In
  // that case simply do not use target info in the compilation
  // because it can be an off-tree target not registered at
  // this point (read: TCE).
  if (!TheTarget || TheTarget->getName() == std::string("cpp")) {
    return 0;
  }

#ifdef LLVM_OLDER_THAN_6_0
  TargetMachine *TM = TheTarget->createTargetMachine(
      TheTriple.getTriple(), MCPU, StringRef(""), GetTargetOptions(),
      Reloc::PIC_, CodeModel::Default, CodeGenOpt::Aggressive);
#else
  TargetMachine *TM = TheTarget->createTargetMachine(
      TheTriple.getTriple(), MCPU, StringRef(""), GetTargetOptions(),
      Reloc::PIC_, CodeModel::Small, CodeGenOpt::Aggressive);
#endif

  assert(TM != NULL && "llvm target has no targetMachine constructor");
  if (device->ops->init_target_machine)
    device->ops->init_target_machine(device->data, TM);
  targetMachines[device] = TM;

  return TM;
}
/* helpers copied from LLVM opt END */

static PassManager &
kernel_compiler_passes(cl_device_id device, llvm::Module *input,
                       const std::string &module_data_layout) {

  PassManager *Passes = nullptr;
  PassRegistry *Registry = nullptr;

  if (kernelPasses.find(device) != kernelPasses.end()) {
    return *kernelPasses[device];
  }

  bool SPMDDevice = device->spmd;

  Registry = PassRegistry::getPassRegistry();

  Triple triple(device->llvm_target_triplet);

  Passes = new PassManager();

  // Need to setup the target info for target specific passes. */
  TargetMachine *Machine = GetTargetMachine(device);

#ifdef LLVM_OLDER_THAN_3_7
  // Add internal analysis passes from the target machine.
  if (Machine)
    Machine->addAnalysisPasses(*Passes);
#else
  if (Machine)
    Passes->add(
        createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
#endif

  if (module_data_layout != "") {
#if (defined LLVM_OLDER_THAN_3_7)
    Passes->add(new DataLayoutPass());
#endif
  }

  /* Disables automated generation of libcalls from code patterns.
     TCE doesn't have a runtime linker which could link the libs later on.
     Also the libcalls might be harmful for WG autovectorization where we
     want to try to vectorize the code it converts to e.g. a memset or
     a memcpy */
#ifdef LLVM_OLDER_THAN_3_7
  TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
  TLI->disableAllFunctions();
  Passes->add(TLI);
#else
  TargetLibraryInfoImpl TLII(triple);
  TLII.disableAllFunctions();
  Passes->add(new TargetLibraryInfoWrapperPass(TLII));
#endif

  /* The kernel compiler passes to run, in order.

     Notes about the kernel compiler phase ordering:

     -mem2reg first because we get unoptimized output from Clang where all
     variables are allocas. Avoid context saving the allocas and make the
     more readable by calling -mem2reg at the beginning.

     -implicit-cond-barriers after -implicit-loop-barriers because the latter
     can inject barriers to loops inside conditional regions after which the
     peeling should be avoided by injecting the implicit conditional barriers.

     -loop-barriers, -barriertails, and -barriers should be ran after the
     implicit barrier injection passes so they "normalize" the implicit
     barriers also.

     -phistoallocas before -workitemloops as otherwise it cannot inject context
     restore code (PHIs need to be at the beginning of the BB and so one cannot
     context restore them with non-PHI code if the value is needed in another
     PHI). */

  std::vector<std::string> passes;
  passes.push_back("remove-optnone");
  passes.push_back("optimize-wi-func-calls");
  passes.push_back("handle-samplers");
  passes.push_back("workitem-handler-chooser");
  passes.push_back("mem2reg");
  passes.push_back("domtree");
  if (device->autolocals_to_args)
    passes.push_back("automatic-locals");

  if (SPMDDevice) {
    passes.push_back("flatten-inline-all");
    passes.push_back("always-inline");
  }  else {
    passes.push_back("flatten-globals");
    passes.push_back("always-inline");
#ifndef LLVM_3_9
    passes.push_back("inline");
#endif
  }

#ifndef LLVM_OLDER_THAN_4_0
  // It should be now safe to run -O3 over the single work-item kernel
  // as the barrier has the attributes preventing illegal motions and
  // duplication. Let's do it to clean up the code for later passes.
  // Especially the WI context structures get needlessly bloated in case there
  // is dead code lying around.
  passes.push_back("STANDARD_OPTS");
#else
  // Just clean up any unused globals.
  passes.push_back("globaldce");
#endif

  if (!SPMDDevice) {
    passes.push_back("simplifycfg");
    passes.push_back("loop-simplify");
    passes.push_back("uniformity");
    passes.push_back("phistoallocas");
    passes.push_back("isolate-regions");
    passes.push_back("implicit-loop-barriers");
    passes.push_back("implicit-cond-barriers");
    passes.push_back("loop-barriers");
    passes.push_back("barriertails");
    passes.push_back("barriers");
    passes.push_back("isolate-regions");
    passes.push_back("wi-aa");
    passes.push_back("workitemrepl");
    //passes.push_back("print-module");
    passes.push_back("workitemloops");
    // Remove the (pseudo) barriers.   They have no use anymore due to the
    // work-item loop control taking care of them.
    passes.push_back("remove-barriers");
  }
  // Add the work group launcher functions and privatize the pseudo variable
  // (local id) accesses.
  if (device->workgroup_pass)
    passes.push_back("workgroup");

  // Attempt to move all allocas to the entry block to avoid the need for
  // dynamic stack which is problematic for some architectures.
  passes.push_back("allocastoentry");

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  // Convert the semantical OpenCL address space IDs to the ones of the target.
  passes.push_back("target-address-spaces");
#endif

  // Later passes might get confused (and expose possible bugs in them) due to
  // UNREACHABLE blocks left by repl. So let's clean up the CFG before running
  // the standard LLVM optimizations.
  passes.push_back("simplifycfg");

#if 0
  passes.push_back("print-module");
  passes.push_back("dot-cfg");
#endif

  if (currentWgMethod == "loopvec" && SPMDDevice)
    passes.push_back("scalarizer");

  passes.push_back("instcombine");
  passes.push_back("STANDARD_OPTS");
  passes.push_back("instcombine");

  // Now actually add the listed passes to the PassManager.
  for (unsigned i = 0; i < passes.size(); ++i) {
    // This is (more or less) -O3.
    if (passes[i] == "STANDARD_OPTS") {
      PassManagerBuilder Builder;
      Builder.OptLevel = 3;
      Builder.SizeLevel = 0;

      // These need to be setup in addition to invoking the passes
      // to get the vectorizers initialized properly.
      if (currentWgMethod == "loopvec") {
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
      }
      Builder.populateModulePassManager(*Passes);
      continue;
    }

    const PassInfo *PIs = Registry->getPassInfo(StringRef(passes[i]));
    if (PIs) {
      // std::cout << "-"<<passes[i] << " ";
      Pass *thispass = PIs->createPass();
      Passes->add(thispass);
    } else {
      std::cerr << "Failed to create kernel compiler pass " << passes[i]
                << std::endl;
      POCL_ABORT("FAIL");
    }
  }

  kernelPasses[device] = Passes;
  return *Passes;
}

void pocl_destroy_llvm_module(void *modp) {

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  llvm::Module *mod = (llvm::Module *)modp;
  if (mod) {
    delete mod;
    --numberOfIRs;
  }
}

// Defined in llvmopencl/WorkitemHandler.cc
namespace pocl {
extern size_t WGLocalSizeX;
extern size_t WGLocalSizeY;
extern size_t WGLocalSizeZ;
extern bool WGDynamicLocalSize;
}

int pocl_llvm_generate_workgroup_function_nowrite(cl_device_id device,
  cl_kernel kernel, size_t local_x, size_t local_y, size_t local_z, void **output) {

  int device_i = pocl_cl_device_to_index(kernel->program, device);
  assert(device_i >= 0);

  pocl::WGDynamicLocalSize = (local_x == 0 && local_y == 0 && local_z == 0);

  currentPoclDevice = device;

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

#ifdef DEBUG_POCL_LLVM_API
  printf("### calling the kernel compiler for kernel %s local_x %zu "
         "local_y %zu local_z %zu parallel_filename: %s\n",
         kernel->name, local_x, local_y, local_z, parallel_bc_path);
#endif

  llvm::Module *input = NULL;
  if (kernel->program->llvm_irs != NULL &&
      kernel->program->llvm_irs[device_i] != NULL) {
#ifdef DEBUG_POCL_LLVM_API
    printf("### cloning the preloaded LLVM IR\n");
#endif
    llvm::Module *p = (llvm::Module *)kernel->program->llvm_irs[device_i];
#ifdef LLVM_OLDER_THAN_3_8
    input = llvm::CloneModule(p);
#else
    input = (llvm::CloneModule(p)).release();
#endif
  } else {
#ifdef DEBUG_POCL_LLVM_API
    printf("### loading the kernel bitcode from disk\n");
#endif
    char program_bc_path[POCL_FILENAME_LENGTH];
    pocl_cache_program_bc_path(program_bc_path, kernel->program, device_i);
    input = parseModuleIR(program_bc_path);
  }

  /* Now finally run the set of passes assembled above */
  // TODO pass these as parameters instead, this is not thread safe!
  pocl::WGLocalSizeX = local_x;
  pocl::WGLocalSizeY = local_y;
  pocl::WGLocalSizeZ = local_z;
  KernelName = kernel->name;

#ifdef LLVM_OLDER_THAN_3_7
  kernel_compiler_passes(device, input,
                         input->getDataLayout()->getStringRepresentation())
      .run(*input);
#else
  kernel_compiler_passes(device, input,
                         input->getDataLayout().getStringRepresentation())
      .run(*input);
#endif

  assert(output != NULL);
  *output = (void *)input;
  ++numberOfIRs;
  return 0;
}


int pocl_llvm_generate_workgroup_function(cl_device_id device, cl_kernel kernel,
                                          size_t local_x, size_t local_y,
                                          size_t local_z) {

  void *modp = NULL;

  int device_i = pocl_cl_device_to_index(kernel->program, device);
  assert(device_i >= 0);

  char parallel_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_work_group_function_path(parallel_bc_path, kernel->program,
                                      device_i, kernel, local_x, local_y,
                                      local_z);

  if (pocl_exists(parallel_bc_path))
    return CL_SUCCESS;

  char final_binary_path[POCL_FILENAME_LENGTH];
  pocl_cache_final_binary_path(final_binary_path, kernel->program, device_i,
                               kernel, local_x, local_y, local_z);

  if (pocl_exists(final_binary_path))
    return CL_SUCCESS;

  int error = pocl_llvm_generate_workgroup_function_nowrite(
      device, kernel, local_x, local_y, local_z, &modp);
  if (error)
    return error;

  error = pocl_cache_write_kernel_parallel_bc(
      modp, kernel->program, device_i, kernel, local_x, local_y, local_z);

  if (error)
    {
      POCL_MSG_ERR ("pocl_cache_write_kernel_parallel_bc()"
                    " failed with %i\n", error);
      return error;
    }

  pocl_destroy_llvm_module(modp);
  return error;
}

int pocl_update_program_llvm_irs(cl_program program,
                                 unsigned device_i) {
  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_program_bc_path(program_bc_path, program, device_i);

  if (!pocl_exists(program_bc_path))
    {
      POCL_MSG_ERR ("%s does not exist!\n",
                     program_bc_path);
      return -1;
    }

  assert(program->llvm_irs[device_i] == nullptr);
  program->llvm_irs[device_i] = parseModuleIR(program_bc_path);
  ++numberOfIRs;
  return 0;
}

void pocl_free_llvm_irs(cl_program program, int device_i) {
  if (program->llvm_irs[device_i]) {
    PoclCompilerMutexGuard lockHolder(NULL);
    InitializeLLVM();
    llvm::Module *mod = (llvm::Module *)program->llvm_irs[device_i];
    delete mod;
    --numberOfIRs;
    program->llvm_irs[device_i] = NULL;
  }
}

void pocl_llvm_update_binaries(cl_program program) {

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  char program_bc_path[POCL_FILENAME_LENGTH];
  int error;

  // Dump the LLVM IR Modules to memory buffers.
  assert(program->llvm_irs != NULL);
#ifdef DEBUG_POCL_LLVM_API
  printf("### refreshing the binaries of the program %p\n", program);
#endif

  for (size_t i = 0; i < program->num_devices; ++i) {
    assert(program->llvm_irs[i] != NULL);
    if (program->binaries[i])
      continue;

    pocl_cache_program_bc_path(program_bc_path, program, i);
    error = pocl_write_module((llvm::Module *)program->llvm_irs[i],
                              program_bc_path, 1);
    assert(error == 0);
    if (error)
      {
        POCL_MSG_ERR ("pocl_write_module(%s) failed!\n",
                     program_bc_path);
        continue;
      }

    std::string content;
    writeModuleIR((llvm::Module *)program->llvm_irs[i], content);

    size_t n = content.size();
    if (n < program->binary_sizes[i])
      POCL_ABORT("binary size doesn't match the expected value");
    if (program->binaries[i])
      POCL_MEM_FREE(program->binaries[i]);
    program->binaries[i] = (unsigned char *)malloc(n);
    std::memcpy(program->binaries[i], content.c_str(), n);

#ifdef DEBUG_POCL_LLVM_API
    printf("### binary for device %zi was of size %zu\n", i,
           program->binary_sizes[i]);
#endif
  }
}


/* Run LLVM codegen on input file (parallel-optimized).
 * modp = llvm::Module* of parallel.bc
 * Output native object file (<kernel>.so.o). */
int pocl_llvm_codegen(cl_kernel kernel, cl_device_id device, void *modp,
                      char **output, size_t *output_size) {

  PoclCompilerMutexGuard lockHolder(NULL);
  InitializeLLVM();

  llvm::Triple triple(device->llvm_target_triplet);
  llvm::TargetMachine *target = GetTargetMachine(device);

  llvm::Module *input = (llvm::Module *)modp;
  assert(input);
  *output = NULL;

  PassManager PM;
#ifdef LLVM_OLDER_THAN_3_7
  llvm::TargetLibraryInfo *TLI = new TargetLibraryInfo(triple);
  PM.add(TLI);
#else
  llvm::TargetLibraryInfoWrapperPass *TLIPass =
      new TargetLibraryInfoWrapperPass(triple);
  PM.add(TLIPass);
#endif
#ifdef LLVM_OLDER_THAN_3_7
  if (target != NULL) {
    target->addAnalysisPasses(PM);
  }
#endif

  // TODO: get DataLayout from the 'device'
  // TODO: better error check
#ifdef LLVM_OLDER_THAN_3_7
  std::string data;
  llvm::raw_string_ostream sos(data);
  llvm::MCContext *mcc;
  if (target && target->addPassesToEmitMC(PM, mcc, sos))
    return 1;
#else
  SmallVector<char, 4096> data;
  llvm::raw_svector_ostream sos(data);
  if (target &&
      target->addPassesToEmitFile(PM, sos, TargetMachine::CGFT_ObjectFile))
    return 1;
#endif

  PM.run(*input);
  std::string o = sos.str(); // flush
  const char *cstr = o.c_str();
  size_t s = o.size();
  *output = (char *)malloc(s);
  *output_size = s;
  memcpy(*output, cstr, s);

  return 0;
}
/* vim: set ts=4 expandtab: */
